# encoding=utf8

import sys
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, support_credentials=True)
#print('Initiating lstm server...', file=sys.stderr)

# Import model
import deep_qa as dq

@app.route("/lstm", methods=['GET', 'OPTIONS'])
@cross_origin(supports_credentials=True)
def get_lstm_response():
	userText = request.args.get('msg')

	best_responses = dq.get_answer(userText)

	if best_responses == False or len(best_responses) <= 0:
		return 'Sorry, I can not help you with that.'
	else:
		response = []
		for answer in best_responses:
			response.append(answer[0])
		return jsonify(response)

if __name__ == "__main__":
	port = 80
	#print('Starting lstm server on port  ' + str(port), file=sys.stderr)
	app.run(host='0.0.0.0', port=port)
	# app.run(port=port)

