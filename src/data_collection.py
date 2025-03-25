from flask import Flask, request, jsonify
from celery import Celery
import time

app = Flask(__name__)

app.config["CELERY_BROKER_URL"] = "redis://localhost:6379/0"
app.config["CELERY_RESULT_BACKEND"] = "redis://localhost:6379/0"
celery = Celery(app.name, broker=app.config["CELERY_BROKER_URL"])
celery.conf.update(app.config)

@celery.task
def collect_user_data(user_id):
    time.sleep(10)
    print(f'Colelcted data for user {user_id}')
    return f'Data collection complete for user {user_id}'
    
@app.route('/collect-data', methods=['POST'])
def collected_data():
    user_id = request.json.get('user_id')
    collect_user_data.delay(user_id)
    return jsonify({'message': 'Data collection started'}), 202
    
if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)