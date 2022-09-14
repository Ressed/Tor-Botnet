docker-compose down
rm -rf tor
docker-compose up -d --scale relay=3 --scale exit=3 --scale client=5
mkdir tor/pcaps