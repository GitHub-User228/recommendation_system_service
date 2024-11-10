#!/bin/bash

export MLFLOW_S3_ENDPOINT_URL=$MLFLOW_S3_ENDPOINT_URL
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
export AWS_BUCKET_NAME=$S3_BUCKET_NAME

mlflow server \
  --registry-store-uri postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME\
  --backend-store-uri postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME\
    --default-artifact-root s3://$AWS_BUCKET_NAME \
    --no-serve-artifacts