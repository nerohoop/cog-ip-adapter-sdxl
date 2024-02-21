### Upload the data set 
RESPONSE=$(curl -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" https://dreambooth-api-experimental.replicate.com/v1/upload/data.zip)

curl -X PUT -H "Content-Type: application/zip" --upload-file data.zip "$(jq -r ".upload_url" <<< "$RESPONSE")"

SERVING_URL=$(jq -r ".serving_url" <<< $RESPONSE)

### Dreambooth training 
### max_train_steps: what could be a good number?
curl -X POST \
    -H "Authorization: Token $REPLICATE_API_TOKEN" \
    -d '{
          "input": {
              "instance_prompt": "a photo of a tok cat",
              "class_prompt": "a photo of a cat",
              "instance_data": "'"$SERVING_URL"'",
              "max_train_steps": 2000
          },
          "model": "nerohoop/dreambooth-tally",
          "trainer_version": "cd3f925f7ab21afaef7d45224790eedbb837eeac40d22e8fefe015489ab644aa",
          "webhook_completed": "https://example.com/dreambooth-webhook"
        }' \
    https://dreambooth-api-experimental.replicate.com/v1/trainings

### SDXL Lora training
curl -s -X POST \
  -H "Authorization: Token r8_HRa8Iux56wBmPlvfAujGEilgg3YMTTQ0tYkTz" \
  -H 'Content-Type: application/json' \
  -d '{
        "destination": "nerohoop/sdxl-fine-tuning-tally", 
        "input": {
          "is_lora": false, 
          "input_images": "https://replicate.delivery/pbxt/KRFrQmVc2wEQgPxAHNufr1FXv8aMQImD8yKbFq69BBgeE8GE/data.zip", 
          "mask_target_prompts": "photo of cat"
        }
      }' \
  https://api.replicate.com/v1/models/stability-ai/sdxl/versions/39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b/trainings


