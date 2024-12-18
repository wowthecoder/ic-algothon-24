import pandas as pd
import numpy as np
import cryptpandas as crp
import json
import re
from flask import Flask, request, jsonify
from slackeventsapi import SlackEventAdapter
from main1 import process_data, calculate_weights_with_constraints, validate_constraints, save_submission
import implemented
import requests

app = Flask(__name__)

# Constants
DATA_FOLDER = "./data_releases"
TEAM_NAME = "limoji"
PASSCODE = "014ls434>"
SUBMISSION_FILE = "submissions1.txt"
SLACK_CLIENT_ID = "8020284472341.8039893431250"
SLACK_CLIENT_SECRET = "1ac9f7fe408aa41eabcf2267caecbbb1"
SLACK_SIGNING_SECRET = "ed6cab16974fec5c811e6a26c6436af8"
CORRECT_JOE_USER_ID = "U080GCRATP1"

slack_events_adapter = SlackEventAdapter(SLACK_SIGNING_SECRET, endpoint="/slack/events")

#setup
with open("algothon_google_api.json") as f:
    google_api_credentials = json.load(f)["installed"]

@app.route('/')
def test():
    return "hello world"

google_form_sent = []

google_url_to_open = ""

@app.route("/url-to-open", methods=['GET'])
def url_to_open_endpoint():
    global google_url_to_open
    return google_url_to_open

@app.route('/message', methods=['POST'])
def messagesendpoint():
    global google_form_sent
    global google_url_to_open


    latest_release = None 
    latest_password = None
    message = request.get_json()["event"]
    print(message, flush=True)
    # If the incoming message contains :passcode" and the user id is the correct Joe
    if 'text' in message and "passcode" in message['text'] and message["user"] == CORRECT_JOE_USER_ID:
        match = re.search(r"the passcode is '([^']+)'", message['text'])
        if match:
            latest_password = match.group(1)
        match = re.search(r"release_(\d+)\.crypt", message['text'])
        # Check if a match is found
        if match:
            latest_release = match.group(1)  # Extract the first capturing group

        if not latest_password or not latest_release:
            return 
        
        # Process the latest file
        latest_file_path = f"{DATA_FOLDER}/release_{latest_release}.crypt"
        data = process_data(latest_file_path, latest_password)

        weights = implemented.get_weights(data)


        # Validate constraints
        # abs_sum_ok, max_abs_ok = validate_constraints(weights)
        # if not abs_sum_ok or not max_abs_ok:
        #     raise ValueError(
        #         f"Validation failed: abs_sum_ok={abs_sum_ok}, max_abs_ok={max_abs_ok}. "
        #         f"Check the weights: {weights.to_dict()}"
        #     )

        # Prepare submission dictionary
        # Prepare submission dictionary
        submission = {
            **weights.to_dict()["weights"],
            **{
                "team_name": TEAM_NAME,
                "passcode": PASSCODE,
            }
        }


        # Output results to terminal
        print("\n\n\n")
        print(f"Submission: {submission}")

        if latest_release not in google_form_sent:
            google_form_url = f"https://docs.google.com/forms/d/e/1FAIpQLSeUYMkI5ce18RL2aF5C8I7mPxF7haH23VEVz7PQrvz0Do0NrQ/formResponse?entry.1985358237={json.dumps(submission)}&emailAddress=lissanartist@gmail.com"
            # response = requests.get(google_form_url)
            # print(response)
            google_url_to_open = google_form_url

            google_form_sent.append(latest_release)

        # Save submission to file
        save_submission(submission, SUBMISSION_FILE, latest_release)

    return ""

# b'{"token":"9CRAS865JOqSoMfPHIBRCf7E","team_id":"T080L8CDWA1","context_team_id":"T080L8CDWA1","context_enterprise_id":null,"api_app_id":"A0815S9CP7C","event":{"user":"U080PUF491B","type":"message","ts":"1731772515.498319","client_msg_id":"3E83E933-49FD-42FC-BB32-3F53194106BA","text":"Got me","team":"T080L8CDWA1","blocks":[{"type":"rich_text","block_id":"3cYse","elements":[{"type":"rich_text_section","elements":[{"type":"text","text":"Got me"}]}]}],"channel":"C080P6M4DKL","event_ts":"1731772515.498319","channel_type":"channel"},"type":"event_callback","event_id":"Ev08161YKVBM","event_time":1731772515,"authorizations":[{"enterprise_id":null,"team_id":"T080L8CDWA1","user_id":"U0818EXG5FE","is_bot":true,"is_enterprise_install":false}],"is_ext_shared_channel":false,"event_context":"4-eyJldCI6Im1lc3NhZ2UiLCJ0aWQiOiJUMDgwTDhDRFdBMSIsImFpZCI6IkEwODE1UzlDUDdDIiwiY2lkIjoiQzA4MFA2TTRES0wifQ"}'


# Example responder to greetings
@slack_events_adapter.on("message")
def idk_whatisthisfor(event_data):
    return "huh"

slack_events_adapter.start(port=8987)