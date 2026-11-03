#!/usr/bin/env python
"""KGNMDA worker (Python 3.6, TensorFlow 1.13.1, Keras 2.3.0; CPU).

Runs the OFFICIAL KGNMDA model (`models.KGCN`, `layers`, `callbacks`, `config.ModelConfig`, `main.get_optimizer`,
`run.read_entity2id_file`, `run.read_kg` from ChangzhiJiang/KGNMDA_master, unmodified) on jobs written by make_jobs.py.

What differs from `run.py` / `main.train` (and only this):
  * data source: folds, similarity kernels and negatives come from the job file (training fold only), not from
    `approved_example.txt` + `generate_gaussian_file`;
  * after training, ALL microbe x disease pairs are scored (the audit ranks held-out positives against every unknown pair)
    instead of scoring one balanced test fold;
  * the pre-sigmoid logit is saved as well as the probability (rank-equivalent; avoids float32 saturation ties).
Hyper-parameters are the repo's `run.py` values (embed 32, depth 2, 8 sampled neighbours, concat aggregator, l2 0.1,
Adam 1e-3, batch 32, <=50 epochs, checkpoint + early stopping (patience 5) on TRAIN AUC). Nothing is tuned.

usage: kgnmda_worker.py prep   <out_dir> <seed,...>                 # build the sampled KG neighbourhoods (official read_kg)
       kgnmda_worker.py run    <jobs_dir> <out_dir> <shard> <nshard> <threads> [dataset]
"""
import os, sys, time, gc, glob, json
VENDOR = '/root/autodl-tmp/mkg/vendor/KGNMDA'
sys.path.insert(0, VENDOR)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np

mode = sys.argv[1]
RAW = os.path.join(VENDOR, 'raw_data', 'mdkg_hmdad')


def prep(out_dir, seeds):
    import run as official                        # official data-processing functions
    os.makedirs(out_dir, exist_ok=True)
    for seed in seeds:
        entity_vocab, relation_vocab = {}, {}
        official.read_entity2id_file(os.path.join(RAW, 'entity2id.txt'), entity_vocab)
        np.random.seed(seed)                      # run.py leaves neighbour sampling unseeded; we fix it per audit seed
        adj_e, adj_r = official.read_kg(os.path.join(RAW, 'train2id.txt'), entity_vocab, relation_vocab, 8)
        np.savez_compressed(os.path.join(out_dir, 'adj_seed%d.npz' % seed), adj_entity=adj_e, adj_relation=adj_r,
                            n_relations=len(relation_vocab))
        print('seed', seed, 'entities', len(entity_vocab), 'relations', len(relation_vocab), 'adj', adj_e.shape, flush=True)


def run(jobs_dir, out_dir, shard, nshard, threads, dataset):
    import tensorflow as tf
    from keras import backend as K
    from keras.models import Model
    import main as official_main                  # get_optimizer
    from models import KGCN
    from config import ModelConfig
    jobs_dir, out_dir = os.path.abspath(jobs_dir), os.path.abspath(out_dir)   # resolve BEFORE chdir below
    os.makedirs(out_dir, exist_ok=True)
    work = os.path.join(out_dir, 'work_shard%d' % shard)
    os.makedirs(os.path.join(work, 'log'), exist_ok=True)
    files = sorted(glob.glob(os.path.join(jobs_dir, dataset + '_seed*.npz')))
    assert files, 'no job files found in %s for %s' % (jobs_dir, dataset)
    mine = [f for i, f in enumerate(files) if i % nshard == shard]
    print('jobs total %d, this shard %d (%d/%d)' % (len(files), len(mine), shard, nshard), flush=True)
    os.chdir(work)                                # callbacks write log/train_history.txt relative to cwd
    adj_cache = {}
    for jf in mine:
        name = os.path.basename(jf)[:-4]
        outf = os.path.join(out_dir, name + '.npz')
        if os.path.exists(outf):
            print('skip', name, flush=True); continue
        J = np.load(jf)
        seed, fold, protocol = int(J['seed']), int(J['fold']), str(J['protocol'])
        n_ent = int(J['n_entities'])
        K.clear_session(); gc.collect()
        tf.set_random_seed(seed * 1000 + fold * 10 + (0 if protocol == 'A' else 1))
        np.random.seed(seed * 1000 + fold * 10 + (0 if protocol == 'A' else 1))
        K.set_session(tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=threads, inter_op_parallelism_threads=1)))
        if seed not in adj_cache:
            A = np.load(os.path.join(out_dir, 'adj_seed%d.npz' % seed))
            ae, ar = A['adj_entity'], A['adj_relation']
            pad = n_ent - ae.shape[0]               # isolated pseudo-entities (microbes absent from the KG): zero rows, as read_kg does
            adj_cache[seed] = (np.vstack([ae, np.zeros((pad, ae.shape[1]), ae.dtype)]), np.vstack([ar, np.zeros((pad, ar.shape[1]), ar.dtype)]), int(A['n_relations']))
        adj_e, adj_r, n_rel = adj_cache[seed]
        micro_eid, dis_eid = J['micro_eid'], J['dis_eid']
        km, kd = J['km'], J['kd']
        nm, nd = len(micro_eid), len(dis_eid)

        cfg = ModelConfig()
        cfg.neighbor_sample_size = 8; cfg.embed_dim = 32; cfg.n_depth = 2; cfg.l2_weight = 1e-1; cfg.dataset = 'mdkg_hmdad'
        cfg.K_Fold = fold + 1; cfg.lr = 1e-3; cfg.optimizer = official_main.get_optimizer('adam', 1e-3); cfg.batch_size = 32
        cfg.aggregator_type = 'concat'; cfg.n_epoch = 50; cfg.callbacks_to_add = ['modelcheckpoint', 'earlystopping']
        cfg.entity_vocab_size = n_ent; cfg.relation_vocab_size = n_rel
        cfg.adj_entity = adj_e; cfg.adj_relation = adj_r
        cfg.disease_pre_feature = np.zeros((n_ent, nd), dtype='float64')     # = generate_pre_embedding(...)
        cfg.microbe_pre_feature = np.zeros((n_ent, nm), dtype='float64')
        cfg.disease_pre_feature[dis_eid] = kd                                  # Gaussian similarity vectors GD, training fold only
        cfg.microbe_pre_feature[micro_eid] = km                                # Gaussian similarity vectors GM, training fold only
        cfg.exp_name = 'kgcn_%s' % name
        cfg.checkpoint_dir = os.path.join(work, 'ckpt'); os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        pos, neg = J['train_pos'], J['train_neg']                              # columns: (microbe idx, disease idx)
        tr = np.vstack([np.column_stack([dis_eid[pos[:, 1]], micro_eid[pos[:, 0]], np.ones(len(pos), int)]),
                        np.column_stack([dis_eid[neg[:, 1]], micro_eid[neg[:, 0]], np.zeros(len(neg), int)])])
        tr = tr[np.random.permutation(len(tr))]
        t0 = time.time()
        model = KGCN(cfg)
        model.fit(x_train=[tr[:, :1], tr[:, 1:2]], y_train=tr[:, 2:3])         # official fit: train-AUC monitor, checkpoint, early stopping
        model.load_best_model()
        train_seconds = time.time() - t0
        dd, mm = np.meshgrid(np.arange(nd), np.arange(nm))                     # all pairs, microbe-major [nm, nd]
        x = [dis_eid[dd.ravel()].reshape(-1, 1), micro_eid[mm.ravel()].reshape(-1, 1)]
        prob = model.model.predict(x, batch_size=2048).reshape(nm, nd)
        logit_model = Model(model.model.inputs, model.model.get_layer('diseasemicrobe_score').output)
        logit = logit_model.predict(x, batch_size=2048).sum(axis=1).reshape(nm, nd)
        hist = [l for l in open('log/train_history.txt')] if False else []
        np.savez_compressed(outf, prob=prob.astype('float32'), logit=logit.astype('float64'), seed=seed, fold=fold, protocol=protocol,
                            train_seconds=train_seconds, n_train=len(tr))
        print('DONE', name, 'train_sec=%.1f' % train_seconds, 'n_train', len(tr), flush=True)
        del model, logit_model


if mode == 'prep':
    prep(sys.argv[2], [int(s) for s in sys.argv[3].split(',')])
else:
    run(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7] if len(sys.argv) > 7 else 'HMDAD')
