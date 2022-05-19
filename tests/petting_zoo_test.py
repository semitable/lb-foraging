import pytest
import pettingzoo.test as pzt
import lbforaging.petting_zoo as lbf
import supersuit as ss
from fst.envs.agent_dict_concat_vecenv import agent_dict_concat_vec_env_v0

def max_cycles_test():
    pzt.max_cycles_test(lbf)

def seed_test():
    pzt.seed_test(lbf.env)

def api_test():
    env = lbf.env()
    pzt.api_test(env, num_cycles=1000)

def parallel_api_test():
    p_env = lbf.parallel_env()
    pzt.parallel_api_test(p_env, num_cycles=1000)

def vec_env_test():
    p_env = lbf.parallel_env()
    mve = ss.pettingzoo_env_to_vec_env_v1(p_env)

def concat_vec_env_test():
    p_env = lbf.parallel_env()
    mve = ss.pettingzoo_env_to_vec_env_v1(p_env)
    cve = ss.concat_vec_envs_v1(mve, 4)
