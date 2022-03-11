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

class Database:
  def __init__(self, meeting_name):
    self.firebase = pyrebase.initialize_app(firebaseConfig)
    self.db = self.firebase.database()
    self.meeting_name = meeting_name
    # auth=firebase.auth()
    # storage=firebase.storage()

  def send(self,message : dict):
    self.db.child(self.meeting_name).child('metrics').push(message)
    print('Pushed to database')

