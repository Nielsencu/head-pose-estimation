import pyrebase

firebaseConfig = {
  "apiKey": "AIzaSyCm8f6gJtK3zvxrXrwg4zMpaRQNmTiZ7UQ",
  "authDomain": "attentionindex.firebaseapp.com",
  "databaseURL": "https://attentionindex-default-rtdb.firebaseio.com",
  "projectId": "attentionindex",
  "storageBucket": "attentionindex.appspot.com",
  "messagingSenderId": "921432105624",
  "appId": "1:921432105624:web:ad22c868065f717e6e26ed",
  "measurementId": "G-7NW9BM2HG3"
}

firebase = pyrebase.initialize_app(firebaseConfig)

db=firebase.database()
# auth=firebase.auth()
# storage=firebase.storage()