from flask import Flask, request
from flask_restx import Api, Resource, fields
from model import CustomerModel
import logging

app = Flask(__name__)
api = Api(app, version='1.0', title='Customer Cluster API', 
          description='A simple Customer Clustering API')

logging.basicConfig(level=logging.DEBUG)

ns = api.namespace('Clustering', description='Operations related to customer clustering')

model = CustomerModel('dataset/loan_default.csv')
model.train_model()
model.assign_labels()

cluster_input_model = api.model('ClusterValue', {
    'loan_amount': fields.Float(required=True, description='The loan amount'),
    'income': fields.Float(required=True, description='Income of the applicant'),
    'Credit_Score': fields.Float(required=True, description='Credit Score of the applicant')
})

@ns.route('/cluster')
class Cluster(Resource):
    @api.expect(cluster_input_model)
    def post(self):
        """Predict the cluster for a new customer."""
        try:
            data = request.json
            logging.info(f"Data received for prediction: {data}")

            if 'loan_amount' not in data or 'income' not in data or 'Credit_Score' not in data:
                raise ValueError("Missing required fields in request data")

            for field in ['loan_amount', 'income', 'Credit_Score']:
                if not isinstance(data[field], (int, float)):
                    raise ValueError(f"Field '{field}' must be a number")

            cluster_id, persona, fraud_propensity = model.predict(data)

            # Depuração - Verificar os tipos
            print(f"Cluster ID Type: {type(cluster_id)}")
            print(f"Persona Type: {type(persona)}")
            print(f"Fraud Propensity Type: {type(fraud_propensity)}")

            response = {
                'cluster': cluster_id,
                'persona': persona,
                'fraud_propensity': fraud_propensity
            }

            # Depuração - Imprimir a resposta
            print("Response:")
            print(response)
            print("Response Type:")
            print(type(response)) 

            return response, 200 

        except ValueError as ve:
            logging.error(f"Validation error: {str(ve)}")
            return {'error': str(ve)}, 400
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return {'error': 'An unexpected error occurred'}, 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)