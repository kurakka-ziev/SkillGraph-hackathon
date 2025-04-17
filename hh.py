from flask import Flask, redirect, request, session, url_for
import requests
import json
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("HH_API_KEY")

CLIENT_ID = 'CLIENT_ID'
CLIENT_SECRET = 'CLIENT_SECRET'
REDIRECT_URI = 'REDIRECT_URI'

@app.route('/')
def index():
    return redirect(f'https://hh.ru/oauth/authorize?response_type=code&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}')

@app.route('/callback')
def callback():
    code = request.args.get('code')
    token_url = 'https://hh.ru/oauth/token'
    data = {
        'grant_type': 'authorization_code',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': code,
        'redirect_uri': REDIRECT_URI
    }
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(token_url, data=data, headers=headers)
    token_info = response.json()
    session['access_token'] = token_info.get('access_token')
    
    if not session['access_token']:
        return 'Ошибка получения access_token', 400

    return redirect(url_for('get_resume'))

@app.route('/resume')
def get_resume():
    access_token = session.get('access_token')
    if not access_token:
        return redirect(url_for('index'))

    headers = {'Authorization': f'Bearer {access_token}'}
    resumes_response = requests.get('https://api.hh.ru/resumes/mine', headers=headers)
    
    if resumes_response.status_code != 200:
        return f"Ошибка получения резюме: {resumes_response.text}", 500

    resumes = resumes_response.json().get('items', [])

    if not resumes:
        return 'Резюме не найдено.'

    resume_id = resumes[0]['id']
    resume_response = requests.get(f'https://api.hh.ru/resumes/{resume_id}', headers=headers)
    
    if resume_response.status_code != 200:
        return f"Ошибка получения данных резюме: {resume_response.text}", 500

    resume_data = resume_response.json()

    skills = resume_data.get('skill_set', [])
    experience = resume_data.get('experience', [])

    parsed_data = {
        'skills': skills,
        'experience': [
            {
                'company': exp.get('company'),
                'position': exp.get('position'),
                'start': exp.get('start'),
                'end': exp.get('end')
            } for exp in experience
        ]
    }

    # отправка навыков на микросервис на порту 5000
    try:
        response = requests.post(
            'http://localhost:5000/process_skills',
            json={'skills': skills},
            headers={'Content-Type': 'application/json'}
        )
        forward_status = response.status_code
        forward_response = response.text
    except Exception as e:
        forward_status = 500
        forward_response = str(e)

    return {
        'parsed_resume': parsed_data,
        'forwarding_status': forward_status,
        'microservice_response': forward_response
    }

if __name__ == '__main__':
    app.run(debug=True, port=5001)
