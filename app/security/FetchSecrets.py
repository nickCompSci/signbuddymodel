from get_docker_secret import get_docker_secret

def fetchSecrets(secret: str):
  secretVariable = get_docker_secret(name=secret, autocast_name=False)
  return secretVariable