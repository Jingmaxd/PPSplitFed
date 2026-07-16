import random
import zlib
import base64
import json
import logging
import gmpy2 as gp
import time

logger = logging.getLogger(__name__)


def _random(maximum, bits):#生成小于maximum的位数为bits的随机数
    rand_function = random.SystemRandom()
    r = gp.mpz(rand_function.getrandbits(bits))
    while r >= maximum:
        r = gp.mpz(rand_function.getrandbits(bits))
    return r

def _random_generator(bits, p, r):#计算h^r mod p,其中h为小于p的位数为bits的随机数
    while True:
        h = _random(p, bits)
        g = gp.powmod(h, r, p)
        if not g == 1:
            break
    return g

def _random_prime(bits):#生成大素数，大于位数为bits的随机数的最接近的素数
    rand_function = random.SystemRandom()
    r = gp.mpz(rand_function.getrandbits(bits))
    r = gp.bit_set(r, bits - 1)
    return gp.next_prime(r)

def _param_generator(bits, r=2):#根据bits生成参数p, q, r，其中p,q均为素数
    while True:
        p = _random_prime(bits)
        q = (p - 1) // 2 
        if gp.is_prime(p) and gp.is_prime(q):
            break
    return p, q, r

def _json_zip(store_dict):
    return base64.b64encode(zlib.compress(
                json.dumps(store_dict).encode('utf-8')
    )).decode('ascii')

def _json_unzip(content):
    try:
        dec_compress = zlib.decompress(base64.b64decode(content.encode('ascii'))).decode('utf-8')
    except Exception:
        raise RuntimeError("Could not decode/unzip the contents")
    try:
        return json.loads(dec_compress)
    except Exception:
        raise RuntimeError("Could interpret the unzipped contents")

#pqrg等参数生成
def generate_config_files(sec_param, sec_param_config):
    TS0 = time.process_time()
    p, q, r = _param_generator(sec_param)
    g = _random_generator(sec_param, p, r)#g=h^r mod p,其中h为小于p的位数为sec_param的随机数
    group_info = {
        'p': gp.digits(p),
        'q': gp.digits(q),
        'r': gp.digits(r)
    }
    sec_param_dict = {'g': gp.digits(g), 'sec_param': sec_param, 'group': group_info} 

    with open(sec_param_config, 'w') as outfile:
        json.dump(sec_param_dict, outfile)
    logger.info('Generate secure parameters config file successfully, see file %s' % sec_param_config)
    TS1 = time.process_time()
    print('sec_param_dict生成时间:%s毫秒' % ((TS1 - TS0)*1000))

def generate_config_files00(sec_param, dlog_table_config, func_bound):
    TS0 = time.process_time()
    p, q, r, g, sec_param = load_sec_param_config('crypto/sec_param_config.json')
    dlog_table = dict()
    bound = func_bound + 1
    for i in range(bound):
        dlog_table[gp.digits(gp.powmod(g, i, p))] = i
    for i in range(-1, -bound, -1):
        dlog_table[gp.digits(gp.powmod(g, i, p))] = i

    dlog_table_dict = {
        'g': gp.digits(g),
        'func_bound': func_bound,
        'dlog_table': dlog_table
    }
    print('计算完成，正在写入')
    with open(dlog_table_config, 'w') as outfile:
        # outfile.write(_json_zip(dlog_table_dict))
        json.dump(dlog_table_dict, outfile)
    logger.info('Generate dlog table config file successfully, see file %s' % dlog_table_config)
    TS1 = time.process_time()
    print('原始小table生成时间:%s毫秒' % ((TS1 - TS0)*1000))
   
def generate_config_files0(sec_param, dlog_table_config0, func_bound):
    TS0 = time.process_time()
    p, q, r, g, sec_param = load_sec_param_config('crypto/sec_param_config.json')
   
    dlog_table0 = dict()
    bound = int(func_bound/2 + 1)
    print('bound',bound)
    for i in range(bound):
        dlog_table0[gp.digits(gp.powmod(g, i, p))] = i
    
    dlog_table_dict0 = {
        'g': gp.digits(g),
        'func_bound': func_bound,
        'dlog_table': dlog_table0
    }
    print('计算完成，正在写入')
    with open(dlog_table_config0, 'w') as outfile:
        # outfile.write(_json_zip(dlog_table_dict))
        json.dump(dlog_table_dict0, outfile)
    logger.info('Generate dlog table config file successfully, see file %s' % dlog_table_config0)
    TS1 = time.process_time()
    print('table0生成时间:%s毫秒' % ((TS1 - TS0)*1000))
    
def generate_config_files1(sec_param, dlog_table_config1, func_bound):
    TS0 = time.process_time()
    p, q, r, g, sec_param = load_sec_param_config('crypto/sec_param_config.json')
   
    dlog_table1 = dict()
    bound = int(func_bound/2 + 1)
    print('bound',bound)
    for i in range(bound,func_bound+1):
        dlog_table1[gp.digits(gp.powmod(g, i, p))] = i
    
    dlog_table_dict1 = {
        'g': gp.digits(g),
        'func_bound': func_bound,
        'dlog_table': dlog_table1
    }
    print('计算完成，正在写入')
    with open(dlog_table_config1, 'w') as outfile:
        # outfile.write(_json_zip(dlog_table_dict))
        json.dump(dlog_table_dict1, outfile)
    logger.info('Generate dlog table config file successfully, see file %s' % dlog_table_config1)
    TS1 = time.process_time()
    print('table1生成时间:%s毫秒' % ((TS1 - TS0)*1000))
    
def generate_config_files2(sec_param, dlog_table_config2, func_bound):
    TS0 = time.process_time()
    p, q, r, g, sec_param = load_sec_param_config('crypto/sec_param_config.json')
 
    dlog_table2 = dict()
    bound = int(func_bound/2 + 1)
    print('bound',bound)
    
    for i in range(-1, -bound, -1):
        dlog_table2[gp.digits(gp.powmod(g, i, p))] = i

    dlog_table_dict2 = {
        'g': gp.digits(g),
        'func_bound': func_bound,
        'dlog_table': dlog_table2
    }
    print('计算完成，正在写入')
    with open(dlog_table_config2, 'w') as outfile:
        # outfile.write(_json_zip(dlog_table_dict))
        json.dump(dlog_table_dict2, outfile)
    logger.info('Generate dlog table config file successfully, see file %s' % dlog_table_config2)
    TS1 = time.process_time()
    print('table2生成时间:%s毫秒' % ((TS1 - TS0)*1000))
    
def generate_config_files3(sec_param, dlog_table_config3, func_bound):
    TS0 = time.process_time()
    p, q, r, g, sec_param = load_sec_param_config('crypto/sec_param_config.json')
 
    dlog_table3 = dict()
    bound = int(func_bound/2 + 1)
    print('bound',bound)
    for i in range( -bound, -func_bound-1,-1):
        dlog_table3[gp.digits(gp.powmod(g, i, p))] = i

    dlog_table_dict3 = {
        'g': gp.digits(g),
        'func_bound': func_bound,
        'dlog_table': dlog_table3
    }
    print('计算完成，正在写入')
    with open(dlog_table_config3, 'w') as outfile:
        # outfile.write(_json_zip(dlog_table_dict))
        json.dump(dlog_table_dict3, outfile)
    logger.info('Generate dlog table config file successfully, see file %s' % dlog_table_config3)
    TS1 = time.process_time()
    print('table3生成时间:%s毫秒' % ((TS1 - TS0)*1000))
    
def load_sec_param_config(sec_param_config_file):
    with open(sec_param_config_file, 'r') as infile:
        sec_param_dict = json.load(infile)

        p = gp.mpz(sec_param_dict['group']['p'])
        q = gp.mpz(sec_param_dict['group']['q'])
        r = gp.mpz(sec_param_dict['group']['r'])
        g = gp.mpz(sec_param_dict['g'])
        sec_param = sec_param_dict['sec_param']

    return p, q, r, g, sec_param

def load_dlog_table_config(dlog_table_config_file):
    with open(dlog_table_config_file, 'r') as infile:
        # config_content = infile.read()
        # store_dict = _json_unzip(config_content)
        store_dict = json.load(infile)

        dlog_table = store_dict['dlog_table']
        func_bound = store_dict['func_bound']
        g = gp.mpz(store_dict['g'])

    return {
        'dlog_table': dlog_table,
        'func_bound': func_bound,
        'g': g
    }


