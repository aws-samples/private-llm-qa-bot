from opensearchpy import OpenSearch

host = 'your-opensearch-host'

from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import boto3

# host = '' # cluster endpoint, for example: my-test-domain.us-east-1.es.amazonaws.com
port = 443
region = 'ap-northeast-1' # e.g. us-west-1

credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, region)
index_name = 'movies'

client = OpenSearch(
    hosts = [f'{host}:{port}'],
    http_auth = auth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection
)

print(client.info())
