import os
import json
import time
from collections import defaultdict

from utils.misc import set_random_seed
from utils.logger import write_to_record_file
from utils.distributed import all_gather, merge_dist_results

from models.vlnbert_init import get_tokenizer

from r2r.agent_cmt_test import Seq2SeqCMTAgentTest

from r2r.data_utils import ImageFeaturesDB, construct_instrs
from r2r.env import R2RBatch
from r2r.parser import parse_args


def build_dataset(args):
    tok = get_tokenizer(args)

    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
    dataset_class = R2RBatch

    train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, ['train'], tokenizer=tok, max_instr_len=args.max_instr_len
    )
    train_env = dataset_class(
        feat_db, train_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
        angle_feat_size=args.angle_feat_size, seed=args.seed,
        sel_data_idxs=None, name='train'
    )

    val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']

    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split], tokenizer=tok, max_instr_len=args.max_instr_len
        )
        val_env = dataset_class(
            feat_db, val_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed,
            sel_data_idxs=None, name=split
        )
        val_envs[split] = val_env

    return train_env, val_envs


def valid(args, train_env, val_envs, p):

    type = f"{p}WEB"
    args.landmark_input = True
    args.p = p
    args.lm_type = "web"

    agent = Seq2SeqCMTAgentTest(args, train_env)
    print("Loaded the listener model at iter %d from %s" % (agent.load(args.resume_file), args.resume_file))

    with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
        json.dump(vars(args), outf, indent=4)
    record_file = os.path.join(args.log_dir, f'valid_{type}.txt')
    write_to_record_file(str(args) + '\n\n', record_file)

    for env_name, env in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        start_time = time.time()
        agent.test(feedback='argmax', iters=iters)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results()
        preds = merge_dist_results(all_gather(preds))

        score_summary, _ = env.eval_metrics(preds)
        loss_str = "Env name: %s" % env_name
        for metric, val in score_summary.items():
            loss_str += ', %s: %.2f' % (metric, val)
        write_to_record_file(loss_str+'\n', record_file)
        break


def main():
    args = parse_args()
    args.anno_dir = '../datasets/RxR/annotations/test/'

    seeds = [6467,4774,325,64675]
    ps = [0, 0.25, 0.5, 0.75]
    for p in ps:
        for seed in seeds:
            args.seed = seed

            set_random_seed(args.seed)
            train_env, val_envs = build_dataset(args)
            valid(args, train_env, val_envs, p)
            time.sleep(5)
            

if __name__ == '__main__':
    main()
