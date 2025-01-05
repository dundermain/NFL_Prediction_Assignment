# NFL_Prediction_Assignment
Multi-Agent System for NFL Team Performance and Winning Probability Analysis using CrewAI

To run:
1. First setup the virtual environment with Python >= 3.10
2. Install crewai and ollama(if you want to run your LLM locally)
3. The knowledge base contains CSV extracted from Kaggle/Profootball and JSON from several free APIs available on SportsRadar. If you want to expand the knowledge base, please add the files in the ‘knowledge’ folder and the path to base_config.yaml under config folder
4. Change the LLM models as per your requirements in line number 16 of crew.py script
5. Then add the corresponding LLM’s API Key in the .env file (if it requires any)
6. Activate the environment and then run the agents with the command ‘crewai run’
7. The launching of CrewAI will open user input field. There you can ask about the games on the terminal

