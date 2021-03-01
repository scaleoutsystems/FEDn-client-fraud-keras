FROM scaleoutsystems/fedn-client:develop
COPY fedn-network.yaml /app/
COPY requirements.txt /app/
WORKDIR /app/

