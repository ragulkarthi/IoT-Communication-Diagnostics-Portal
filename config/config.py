import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    "log_analytics": {
        "workspace_id": "901839b7-5812-4403-98fb-c0610b0c9ac7",
        "client_id": "f5894ab9-822a-4c07-bf41-58b02ccf3483",
        "client_secret": "***********",
        "tenant_id": "2f16a741-bc3a-42ec-831e-fda5267388cf",
    },
    "sumologic": {
        "api_url": "https://api.sumologic.com/api/v1/search/jobs",
        "access_id": "sucQuif3KVo0mk",
        "access_key": "************"
    }
}