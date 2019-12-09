from flask import Flask, jsonify, abort, make_response, request, url_for,Response
from flask_httpauth import HTTPBasicAuth
import pymysql

app = Flask(__name__)
auth = HTTPBasicAuth()

keys = []
db = pymysql.connect(host="localhost",user="root",password="123456",db="kmip")
cursor = db.cursor()
sql = """CREATE TABLE IF NOT EXISTS key_name(
           name varchar(100) NOT NULL,
           length int(10) NOT NULL,
           cipher varchar(50) NOT NULL,
           description varchar(50) DEFAULT NULL,
           version int(10) NOT NULL,
           versionName varchar(10) NOT NULL,
           material varchar(100) NOT NULL
      )ENGINE=InnoDB DEFAULT CHARSET=utf8;
      """
cursor.execute(sql)
sql_query = "select * from key_name"
cursor.execute(sql_query)
if cursor.rowcount == 0:
   pass
else:
   alist = cursor.fetchall()
   for row in alist:
       key_old = {
        "name": row[0],
        "length": row[1],
        "cipher": row[2],
        'description': row[3],
        'version': 1,
        "versionName": "1",
        "material": row[6]
       }
       keys.append(key_old)
db.close()

@app.route('/kms/v1/keys', methods=['POST'])
def create_a_key():
    if not request.json or not 'name' in request.json:
        abort(400)
    # Build the client and connect to the server
    name = request.json["name"]
    key = filter(lambda t: t['name'] == name, keys)
    key = list(key)
    if len(key) == 0:
        from kmip.pie import client
        from kmip.core import enums
        with client.ProxyKmipClient() as client:
            ciphers = {
                'AES/CTR/NoPadding':enums.CryptographicAlgorithm.AES,
                'AES':enums.CryptographicAlgorithm.AES,
                'Blowfish':enums.CryptographicAlgorithm.BLOWFISH,
                'Camellia':enums.CryptographicAlgorithm.CAMELLIA,
                'CAST5':enums.CryptographicAlgorithm.CAST5,
                'DES':enums.CryptographicAlgorithm.DES,
                'IDEA':enums.CryptographicAlgorithm.IDEA,
                'CHACHA20':enums.CryptographicAlgorithm.CHACHA20
            }
            cipher = ciphers.get(request.json['cipher'],None)
            length = request.json['length']
            if cipher == None:
                abort(404)
            uid = client.create(
                cipher,
                length,
                operation_policy_name='default'
                )
            secret = client.get(uid)
            ss = "{0}".format(secret)
        key_new = {
            "name": request.json["name"],
            "length": length,
            "cipher": "AES/CTR/NoPadding",
            'description': request.json.get('description', ""),
            'version': 1,
            "versionName": "1",
            "material": ss[0:22]
            #"material":"123456789abcdef0123456"
        }
        key_ret = {
            "name": request.json["name"],
            "versionName": "1",
            "material": ss[0:22]
            #"material":"123456789abcdef0123456"
        }
        keys.append(key_new)
        mm = ss[0:22]
        con = pymysql.connect(host="localhost",user="root",password="123456",db="kmip")
        sql_insert = "INSERT INTO key_name VALUES('%s','%d','AES/CTR/NoPadding','null',1,'1','%s')" % (name,length,mm)
        #sql_insert = "INSERT INTO key_name VALUES('11',128,'AES/CTR/NoPadding','null',1,'1','12345678')"
        try:
            cu = con.cursor()
            cu.execute(sql_insert)
            con.commit()
        except:
            con.rollback()
            abort(404)
        con.close()
    else:
        key_ret = {
            "name": request.json["name"],
            "versionName": "1",
            "material": key[0]["material"]
            #"material":"123456789abcdef0123456"
        }
    return jsonify(key_ret), 201

@app.route('/kms/v1/key/<string:key_name>', methods=['POST'])
def rollover_key(key_name):
    key = filter(lambda t: t['name'] == key_name, keys)
    key = list(key)
    if len(key) == 0:
        abort(404)
    key_roll = {
        "name": key[0]["versionName"],
        "material":key[0]["material"]
    }
    return jsonify(key_roll), 200

@app.route('/kms/v1/key/<string:key_name>', methods=['DELETE'])
def delete_key(key_name):
    key = filter(lambda t: t['name'] == key_name, keys)
    key = list(key)
    if len(key) == 0:
        abort(404)
    keys.remove(key[0])
    return jsonify(),200

@app.route('/kms/v1/key/<string:key_name>/_metadata', methods=['GET'])
def get_key_metadata(key_name):
    key = filter(lambda t: t['name'] == key_name, keys)
    key = list(key)
    if len(key) == 0:
        abort(404)
    ret = {
        "name" : key[0]['name'],
        "cipher": "AES/CTR/NoPadding",
        "length":key[0]["length"],
        "description": "this",
        "attributes": {
    "key.acl.name" : key[0]['name']
  },
        "created": 1540967779465,
        "versions": 1
    }
    return jsonify(ret), 200

@app.route('/kms/v1/key/<string:key_name>/_currentversion',methods=['GET'])
def get_current_key(key_name):
    key = filter(lambda t: t['name'] == key_name, keys)
    key = list(key)
    if len(key) == 0:
        abort(404)
    ret = {
        #"name":"fff"
        "name":key[0]['versionName'],
        "material":key[0]["material"]
    }
    return jsonify(ret),200



@app.route('/kms/v1/keys/metadata', methods=['GET'])
def get_keys_metadata():
    ret = [ ]
    return jsonify(ret)

@app.route('/kms/v1/key/<string:key_name>/_eek', methods=['GET'])
def get_key_eek(key_name):
    key = filter(lambda t: t['name'] == key_name, keys)
    key = list(key)
    if len(key) == 0:
        abort(404)
    ret = [
        {
    "versionName" : "encryptionVersionName",
    "iv"                  : "123456789abcdef0123456",
    "encryptedKeyVersion" : {
        "versionName"       : "EEK",
        "material"          : key[0]["material"]
    }
  }
    ]
    print ret
    print len(ret[0]["iv"])
    return jsonify(ret)

@app.route('/kms/v1/keyversion/encryptionVersionName/_eek', methods=['POST'])
def get_key_dek():
    key_name = request.json['name']
    key = filter(lambda t: t['name'] == key_name, keys)
    key = list(key)
    if len(key) == 0:
        abort(404)
    ret = {
     "name" : "EK",
     "material"          : key[0]["material"]
    }
    return jsonify(ret)

@app.route('/kms/v1/keys/names', methods=['GET'])
def get_key_names():
    key_names = []
    for key in keys:
        key_names.append(key['name'])
    return jsonify(key_names)  #if there is a ret,the it will have next OPTIONS requst

if __name__ == '__main__':
    app.run(host = '10.10.4.122', port=5000, debug = True)
