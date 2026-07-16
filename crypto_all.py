import logging
from utils import _random, load_sec_param_config,load_dlog_table_config
import gmpy2 as gp
import numpy as np
import math


logger = logging.getLogger(__name__)
precision_data=3
precision_server=3
precision_server_mife=1

eta = 3200 #MLP
#eta = 205000 #CNN

#system parameters
p, q, r, g, sec_param = load_sec_param_config('crypto/sec_param_config.json')
#for MLP model, one small dlog table is enough
#use generate_config_files00 in utils.py to generate dlog_table_config.json
dlog=load_dlog_table_config('crypto/dlog_table_config.json')
dlog_table=dlog['dlog_table']

#for CNN model, use big dlog table
#dlog0=load_dlog_table_config('crypto/dlog_table_config0.json')
#dlog_table0=dlog0['dlog_table']
#print('table0')
#dlog1=load_dlog_table_config('crypto/dlog_table_config1.json')
#dlog_table1=dlog1['dlog_table']
#print('table1')
#dlog2=load_dlog_table_config('crypto/dlog_table_config2.json')
#dlog_table2=dlog2['dlog_table']
#print('table2')
#dlog3=load_dlog_table_config('crypto/dlog_table_config3.json')
#dlog_table3=dlog3['dlog_table']
#print('table3')

#sife
msk = [_random(p, sec_param) for i in range(eta)]
pk = [gp.powmod(g, msk[i], p) for i in range(eta)]
mpk = {'p': p, 'g': g, 'pk': pk}
print('运行crypto_sife')

#mife
parties = {
    'id_1': 1,
    'id_2': 1,
    'id_3': 1,
    'id_4': 1,
    'id_5': 1
}
w, u, g_w = dict(), dict(), dict()
for idx in parties.keys():
    w_idx, u_idx, g_w_idx = list(), list(), list()
    rnd = _random(p, sec_param)
    w_idx.append(rnd)
    g_w_idx.append(gp.powmod(g, rnd, p))
    u_idx.append(_random(p, sec_param))
    w[idx] = w_idx
    u[idx] = u_idx
    g_w[idx] = g_w_idx
    
m_mpk={'p': p, 'g': g, 'sec_param': sec_param,'g_w': g_w,'parties': parties}
m_msk = {'w': w, 'u': u}
print('运行crypto_mife')


def generate_common_public_key():
    pk = dict()
    pk['g'] = gp.digits(mpk['g'])
    pk['p'] = gp.digits(mpk['p'])
    return pk

def generate_public_key(vec_size):
    assert vec_size <= eta
    pk = dict()
    pk['bound'] = vec_size
    pk['g'] = gp.digits(mpk['g'])
    pk['p'] = gp.digits(mpk['p'])
    pk['pk'] = list()
    for i in range(vec_size):
        pk['pk'].append(gp.digits(mpk['pk'][i]))
    return pk    
        
def generate_private_key(vec):
    assert len(vec) <= eta
    sk = gp.mpz(0)
    for i in range(len(vec)):
        sk = gp.add(sk, gp.mul(msk[i], vec[i]))
    return {'bound': len(vec), 'sk': gp.digits(sk)}
        
def encrypt(pk, vec):
        assert len(vec) == pk['bound']

        p = gp.mpz(pk['p'])
        g = gp.mpz(pk['g'])

        r = _random(p, sec_param)
        ct0 = gp.digits(gp.powmod(g, r, p))
        ct_list = []
        for i in range(len(vec)):
            ct_list.append(gp.digits(
                gp.mul(
                    gp.powmod(gp.mpz(pk['pk'][i]), r, p),
                    gp.powmod(g, gp.mpz(int(vec[i])), p)
                )
            ))
        return {'ct0': ct0, 'ct_list': ct_list}

def decrypt(pk, sk, vec, ct, max_innerprod):
        p = gp.mpz(pk['p'])
        g = gp.mpz(pk['g'])

        res = gp.mpz(1)
        for i in range(len(vec)):
            res = gp.mul(
                res,
                gp.powmod(gp.mpz(ct['ct_list'][i]), gp.mpz(vec[i]), p)
            )
        res = gp.t_mod(res, p)
        g_f = gp.divm(res, gp.powmod(gp.mpz(ct['ct0']), gp.mpz(sk['sk']), p), p)
        f = _solve_dlog(p, g, g_f, max_innerprod)

        return f

def execute(data_array):
        data_list = (data_array * pow(10, precision_data)).astype(int).flatten().tolist()
        pk = generate_public_key(len(data_list))
        ct_data = encrypt(pk, data_list)
        return ct_data

def execute_ndarray(data_ndarray):
        assert type(data_ndarray) == np.ndarray, 'input data should be in numpy array format'
        assert len(data_ndarray.shape) == 2, 'at present, only address 2d array'

        ct_list = [execute(data_ndarray[i, :]) for i in range(data_ndarray.shape[0])]
        return ct_list
            
def request_key(data_array):
        data_list = (data_array * pow(10, precision_server)).astype(int).flatten().tolist()
        sk = generate_private_key(data_list)
        return sk

def request_key_ndarray(data_ndarray):
        assert type(data_ndarray) == np.ndarray, 'input weight should be a numpy array'
        assert len(data_ndarray.shape) == 2, 'only address 2d array'

        sk_list = [request_key(data_ndarray[i, :]) for i in range(data_ndarray.shape[0])]
        return sk_list
       
def serverexecute(sk, ct, data_array):
        data_list = (data_array * pow(10, precision_server)).astype(int).flatten().tolist()
        max_inner_prod = 100000000 
        common_pk = generate_common_public_key()
        dec_prod = decrypt(common_pk, sk, data_list, ct, max_inner_prod)

        if dec_prod is None:
            logger.debug('find a bad case - decryption: ')
            #assert False
        return float(dec_prod)/pow(10, precision_server)/pow(10, precision_data)    

def serverexecute_ndarray(sk_list, ct_list, data_ndarray):
        assert type(data_ndarray) == np.ndarray, 'input weight should be a numpy array'
        assert len(data_ndarray.shape) == 2, 'only address 2d array'
        assert len(sk_list) == data_ndarray.shape[0]

        res = np.zeros((data_ndarray.shape[0], len(ct_list)))
        for i in range(data_ndarray.shape[0]):
            for j in range(len(ct_list)):
                res[i][j] = serverexecute(sk_list[i], ct_list[j], data_ndarray[i, :])
        return res 

def m_generate_common_public_key():
    g_w = m_mpk['g_w']
    #print('g_w ',g_w )
    g_w_digits = dict()
    for idx in g_w.keys():
        g_w_digits[idx] = [gp.digits(j) for j in g_w[idx]]
    return {
        'g': gp.digits(m_mpk['g']),
        'p': gp.digits(m_mpk['p']),
        'sec_param': m_mpk['sec_param'],
        'g_w': g_w_digits
    }

def m_generate_public_key(slot_index):
    slot_mpk = m_generate_common_public_key()

    if slot_index in m_msk['w'] and slot_index in m_msk['u']:
        slot_mpk['w'] = [gp.digits(i) for i in m_msk['w'][slot_index]]
        slot_mpk['u'] = [gp.digits(i) for i in m_msk['u'][slot_index]]
        
        return slot_mpk
    else:
        logger.error('the slot id is not found.')
        return None

def _gen_total_parties_vec_size( parties):
    count = 0
    for idx in parties.keys():
        count = count + parties[idx]
    return count

def _split_vector(vec, parties):
    vec_split = dict()
    i = 0
    for idx in parties.keys():
        vec_split[idx] = vec[i: i+parties[idx]]
        i = i + parties[idx]

    return vec_split

def m_generate_private_key(vec, parties):
    vec_parties = _split_vector(vec, parties)

    d = dict()
    z = gp.mpz(0)
    for idx in parties.keys():
        if idx in m_msk['w'] and idx in m_msk['u']:
            w_idx = m_msk['w'][idx]
            u_idx = m_msk['u'][idx]
            vec_idx = vec_parties[idx]
            vec_w_idx = gp.mpz(0)
            vec_w_idx = vec_w_idx + gp.mul(gp.mpz(vec_idx[0]), w_idx[0])
            d[idx] = gp.digits(vec_w_idx)
            z = z + gp.mul(gp.mpz(vec_idx[0]), u_idx[0])
        else:
            logger.error('the id %s in parties is not found.' % idx)

    return {'d': d, 'z': gp.digits(z)}

def m_encrypt( slot_pk, vec):
    p = gp.mpz(slot_pk['p'])
    g = gp.mpz(slot_pk['g'])
    sec_param = slot_pk['sec_param']
    u = slot_pk['u']
    w = slot_pk['w']
    r = _random(p, sec_param)
    c0 = gp.digits(gp.powmod(g, r, p))

    c = [gp.digits(gp.powmod(g, gp.mpz(vec[i]) + gp.mpz(u[0]) + gp.mul(gp.mpz(w[0]), r), p)) for i in range(len(vec))]
    return {'t': c0, 'c': c}

def m_decrypt( common_pk_mife, sk, vec, ct, max_inner_prod):
    assert len(ct['ct_dict']) == len(sk['d'])
    p = gp.mpz(common_pk_mife['p'])
    g = gp.mpz(common_pk_mife['g'])
    z = gp.mpz(sk['z'])
    d = sk['d']

    vec_parties = _split_vector(vec, ct['parties'])
    ct_dict = ct['ct_dict']

    g_f = list()
    Dec = list()
    Dec0 = list()

    lc=ct_dict['id_1']['c']

    g_f0 = [gp.mpz(1) for i in range(len(lc))]

    for idx in ct_dict.keys():
        vec_idx = vec_parties[idx]
        c_idx = ct_dict[idx]['c']
        t_idx = ct_dict[idx]['t']
        d_idx = d[idx]

        Dec0 = [1 for i in range(len(c_idx))]
        for j in range(len(c_idx)):
            init_idx = gp.powmod(gp.mpz(c_idx[j]), gp.mpz(vec_idx[0]), p)

            Dec0[j] = gp.divm(gp.mpz(init_idx), gp.powmod(gp.mpz(t_idx), gp.mpz(d_idx), p), p)
                   
        g_f0 =np.multiply(g_f0 , Dec0)

    
    g_f = [gp.digits(gp.divm(gp.mpz(g_f0[i]), gp.powmod(g, z, p), p) ) for i in range(len(g_f0))]
    f = [gp.digits(_solve_dlog(p, g, gp.mpz(g_f[i]), max_inner_prod))for i in range(len(g_f))]
    f=np.array(f, dtype = float)

    return  f

def m_execute(data_array, params):
    data_list = (data_array * pow(10, precision_data)).astype(int).flatten().tolist()
    id = params['id']
    m_pk = m_generate_public_key(id)
    ct_data = m_encrypt(m_pk, data_list)
    return ct_data

def m_execute_ndarray(data_ndarray, params):
    assert type(data_ndarray) == np.ndarray, 'input data should be in numpy array format'
    assert len(data_ndarray.shape) == 2, 'at present, only address 2d array'

    ct_list = [m_execute(data_ndarray[i, :], params) for i in range(data_ndarray.shape[0])]
    return ct_list

def m_request_key( data_array, params):
    sk = None
    data_list = (data_array * pow(10,precision_server_mife)).astype(int).flatten().tolist()
    sk = m_generate_private_key(data_list, params['parties'])
    return sk

def m_request_key_ndarray(data_ndarray, params):
    assert type(data_ndarray) == np.ndarray, 'input weight should be a numpy array'
    assert len(data_ndarray.shape) == 2, 'only address 2d array'

    sk_list = [m_request_key(data_ndarray[i, :], params) for i in range(data_ndarray.shape[0])]
    return sk_list

def m_serverexecute( sk, ct, data_array, params):
    dec_res = None
    data_list = (data_array * pow(10, precision_server_mife)).astype(int).flatten().tolist()
    max_inner_prod = 100000000 
    common_pk_mife = m_generate_common_public_key()
    if params['type'] == 'mife':
        dec_res = m_decrypt(common_pk_mife, sk, data_list, ct, max_inner_prod)
    if dec_res is None:
        logger.debug('find a bad case - decryption: ')
        assert False
    return dec_res/pow(10, precision_server_mife)/pow(10, precision_data)

def m_serverexecute_ndarray( sk_list, ct_list, data_ndarray, params):
    assert type(data_ndarray) == np.ndarray, 'input weight should be a numpy array'
    assert len(data_ndarray.shape) == 2, 'only address 2d array'
    assert len(sk_list) == data_ndarray.shape[0]

    res = np.zeros((data_ndarray.shape[0], len(ct_list)))
    for i in range(data_ndarray.shape[0]):
        for j in range(len(ct_list)):
            res[i][j] = m_serverexecute(sk_list[i], ct_list[j], data_ndarray[i, :], params)
    return res

def _solve_dlog(p, g, h, dlog_max):
        """
        Attempts to solve for the discrete logh where g^x = h mod p via
        hash table.
        """
        if gp.digits(h) in dlog_table:
            return dlog_table[gp.digits(h)]
        #elif gp.digits(h) in dlog_table1:
        #    return dlog_table1[gp.digits(h)]
        #elif gp.digits(h) in dlog_table2:
        #    return dlog_table2[gp.digits(h)]
        #elif gp.digits(h) in dlog_table3:
        #    return dlog_table3[gp.digits(h)]
        else:
            logger.warning("did not find f in dlog table, may cost more time to compute")
            return _solve_dlog_naive(p, g, h, dlog_max)

def _solve_dlog_naive(p, g, h, dlog_max):
        """
        Attempts to solve for the discrete log x, where g^x = h, via
        trial and error. Assumes that x is at most dlog_max.
        """
        res = None
        for j in range(dlog_max):
            if gp.powmod(g, j, p) == gp.mpz(h):
                res = j
                break
        if res == None:
            h = gp.invert(h, p)
            for i in range(dlog_max):
                if gp.powmod(g, i, p) == gp.mpz(h):
                    res = -i
        return res

def _solve_dlog_bsgs(g, h, p):
        """
        Attempts to solve for the discrete log x, where g^x = h mod p,
        via the Baby-Step Giant-Step algorithm.
        """
        m = math.ceil(math.sqrt(p-1)) # phi(p) is p-1, if p is prime
        # store hashmap of g^{1,2,...,m}(mod p)
        hash_table = {pow(g, i, p): i for i in range(m)}
        # precompute via Fermat's Little Theorem
        c = pow(g, m * (p-2), p)
        # search for an equivalence in the table. Giant Step.
        for j in range(m):
            y = (h * pow(c, j, p)) % p
            if y in hash_table:
                return j * m + hash_table[y]

        return None
