import subprocess
import time
i = 0
import json


while True:
    try:

        # Run the script using subprocess
        #process = subprocess.Popen(['python', 'main_chess_upright.py'])
        process = subprocess.Popen(['blenderproc', 'run', 'main_chess_upright.py', 'datasets/models', 'datasets/cc_textures', 'output', '--num_scenes=200'])

        # Wait for the script to complete
        process.wait()
        
    except subprocess.CalledProcessError as e:
        # Handle specific error if needed
        print(f"Script crashed with error: {e.returncode}")
        
    except KeyboardInterrupt:
        # Handle keyboard interrupt (Ctrl+C)
        print("Script stopped by user")
        break
        
    else:
        # Script completed successfully
        print("Script completed")
        
    with open('/home/max/Documents/GitHub/thesis/output/bop_data/lm/train_pbr/000000/scene_camera.json') as yml_file:
        json_data = json.load(yml_file)

    

    if len(json_data) > 65000:
        break
    # Wait for a specific duration before restarting
    # You can adjust the delay as per your requirements
    print("Restarting script...")
    i =  i + 1
    time.sleep(4)
