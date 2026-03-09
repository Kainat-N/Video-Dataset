import json
from interactive_chat import InteractiveChat
from validator import validate_attributes
from config_saver import save_config

API_KEY = "AIzaSyBcmQdOdN-ArYuhu2U3zzEpyHTMWr7ZNtw"

chatbot = InteractiveChat(API_KEY)

print("=== Facial Dataset Configuration Assistant ===\n")

# Start conversation
print("Assistant:", chatbot.send("Start"))

while True:
    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    response = chatbot.send(user_input)
    print("\nAssistant:", response)

    if any(x in user_input.lower() for x in ["yes", "confirm"]):
        # Ask Gemini to output only JSON based on conversation
        json_request_prompt = "Please output the final dataset configuration in valid JSON only."

        json_response = chatbot.send(json_request_prompt)
        try:
            parsed = json.loads(json_response)
            validated = validate_attributes(parsed)
            save_config(validated)
            print("\n✅ Validated configuration saved to dataset_config.json")
        except Exception as e:
            print("\n❌ Failed to parse JSON:", e)
        break

    # # Try parsing JSON (only after confirmation)
    # try:
    #     parsed = json.loads(response)

    #     # Validate
    #     validated = validate_attributes(parsed)

    #     # Save
    #     save_config(validated)

    #     print("\n✅ Validated configuration saved to dataset_config.json")
    #     break

    # except:
    #     pass
