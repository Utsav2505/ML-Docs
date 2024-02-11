
import pickle
loaded_model = pickle.load(open('phishing.pkl', 'rb'))

predict_bad = ['fastapi.tiangolo.com/','youtube.com/watch?v=zKNXHluHneU','girishgr.github.io/NetflixClonePractice'
               ,'yeniy47.top/','mjytn.blogspot.fi/']
result = loaded_model.predict(predict_bad)
print(result)