from cryptography.fernet import Fernet as fe
key=fe.generate_key()
print(key)
ins=fe(key)
m='Hello'.encode()
print(m)
en=ins.encrypt(m)
print(en)
de=ins.decrypt(en)+b'hrllo'
print(de)
dec=de.decode()
print(dec)
