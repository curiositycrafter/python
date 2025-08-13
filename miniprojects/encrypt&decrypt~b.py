from cryptography.fernet import Fernet

key = Fernet.generate_key() # generates a new key in byte format
f=Fernet(key)# create an instance to use its methods 
msg = "10001110110011000111".encode()   # encode converts string -> bytes (encoding)
msgv2 = f.encrypt(msg)
print(f.decrypt(msgv2).decode())  # decode converts bytes->string(decoding)

from datetime import datetime
formatted = datetime.now().strftime("%Y-%m-%d %H:%M:%S")#stringformattime
print("Formatted:", formatted)
