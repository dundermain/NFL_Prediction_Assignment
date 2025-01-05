
import yaml
from .config_generator import create_yaml_config

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import CSVSearchTool, JSONSearchTool, WebsiteSearchTool
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource



create_yaml_config()

# llm = LLM(model = 'ollama/gemma2', base_url = 'http://localhost:11434')
llm = LLM(model="gpt-4o",temperature=0.8,)
# llm = LLM(model="gemini/gemini-1.5-pro",temperature=0.7)



@CrewBase
class NflPredicitonAssistant():
	"""NflPredicitonAssistant crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'
	base_config = 'config/base_config.yaml'

	with open(base_config, "r") as yaml_file:
		base_config_data = yaml.safe_load(yaml_file)


	# Extract JSON paths from the YAML file
	json_paths = base_config_data.get("knowledge_base", {}).get("json_paths", [])
	csv_paths = base_config_data.get("knowledge_base", {}).get("csv_paths", [])
	nfl_websites = list(base_config_data.get("nfl_websites", {}).values())


	json_source = JSONKnowledgeSource(
		# file_paths=json_paths,
		file_path = json_paths,
		collection_name='knowledge',
		metadata={"category": "knowledge_jsons"},
		)
	


	csv_source = CSVKnowledgeSource(
		file_path= csv_paths,
		collection_name='knowledge',
		metadata={"category": "knowledge_csvs"},
		)
	


	#sequence of agents
	@agent
	def trend_analysis_agent(self) -> Agent:

		trend_analysis = Agent(
			config = self.agents_config['trend_analysis_agent'],
			verbose = True,
			llm = llm,
			# tools = [RetrievalTool(config_path=self.base_config)],
			tools = [JSONSearchTool(), CSVSearchTool(), WebsiteSearchTool()],
		)

		print("Trend Analysis Agent Initialised")
		return trend_analysis
	
	@agent
	def team_changes_agent(self) -> Agent:

		team_changes = Agent(
			config = self.agents_config['team_changes_agent'],
			verbose = True,
			llm = llm,
			# tools = [RetrievalTool(config_path=self.base_config)],
			tools = [JSONSearchTool(), CSVSearchTool(), WebsiteSearchTool()],
		)

		print("Team changes Agent Initialised")
		return team_changes
	
	@agent
	def injury_analysis_agent(self) -> Agent:

		injury_analysis = Agent(
			config = self.agents_config['injury_analysis_agent'],
			verbose = True,
			llm = llm,
			# tools = [RetrievalTool(config_path=self.base_config)],
			tools = [JSONSearchTool(), CSVSearchTool(), WebsiteSearchTool()],
		)

		print("Injury Analysis Agent Initialised")
		return injury_analysis
	
	@agent
	def head_to_head_analysis_agent(self) -> Agent:

		head_to_head_analysis = Agent(
			config = self.agents_config['head_to_head_analysis_agent'],
			verbose = True,
			llm = llm,
			# tools = [RetrievalTool(config_path=self.base_config)],
			tools = [JSONSearchTool(), CSVSearchTool(), WebsiteSearchTool()],
		)

		print("Team's head to head Analysis Agent Initialised")
		return head_to_head_analysis

	@agent
	def current_season_performance_agent(self) -> Agent:

		current_season_performance = Agent(
			config = self.agents_config['current_season_performance_agent'],
			verbose = True,
			llm = llm,
			# tools = [RetrievalTool(config_path=self.base_config)],
			tools = [JSONSearchTool(), CSVSearchTool(), WebsiteSearchTool()],
		)

		print("Current Season Performance Analysis Agent Initialised")
		return current_season_performance	

	@agent
	def coaching_strategy_analysis_agent(self) -> Agent:

		coaching_strategy_analysis = Agent(
			config = self.agents_config['coaching_strategy_analysis_agent'],
			verbose = True,
			llm = llm,
			# tools = [RetrievalTool(config_path=self.base_config)],
			tools = [JSONSearchTool(), CSVSearchTool(), WebsiteSearchTool()],
		)

		print("Coaching Stategy Analysis Agent Initialised")
		return coaching_strategy_analysis
	
	@agent
	def environmental_impact_analysis_agent(self) -> Agent:

		environmental_impact_analysis = Agent(
			config = self.agents_config['environmental_impact_analysis_agent'],
			verbose = True,
			llm = llm,
			# tools = [RetrievalTool(config_path=self.base_config)],
			tools = [JSONSearchTool(), CSVSearchTool(), WebsiteSearchTool()],
		)

		print("Environmental Impact Analysis Agent Initialised")
		return environmental_impact_analysis
	
	@agent
	def performance_summary_agent(self) -> Agent:

		performance_summary = Agent(
			config = self.agents_config['performance_summary_agent'],
			verbose = True,
			llm = llm,
		)

		print("Performance Summary Agent Initialised")
		return performance_summary

	@agent
	def consensus_agent(self) -> Agent:

		consensus_agent = Agent(
			config=self.agents_config['consensus_agent'],
			verbose=True,
			llm = llm,
		)
		print("Consensus Summary Agent Initialised")
		return consensus_agent
	

	manager = Agent(
		role="Project Manager",
		goal="Efficiently manage the crew and ensure high-quality task completion",
		backstory="You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
		allow_delegation=True,
		)
	

	#sequence of tasks
	@task
	def trend_analysis_task(self) -> Task:

		trend_analysis_task = Task(
			config = self.tasks_config['trend_analysis_task'],
		)
		return trend_analysis_task

	@task
	def team_changes_task(self) -> Task:

		team_changes_task = Task(
			config = self.tasks_config['team_changes_task'],
		)
		return team_changes_task


	@task
	def injury_analysis_task(self) -> Task:

		injury_analysis_task = Task(
			config = self.tasks_config['injury_analysis_task'],
		)
		return injury_analysis_task
	

	@task
	def head_to_head_analysis_task(self) -> Task:

		head_to_head_analysis_task = Task(
			config = self.tasks_config['head_to_head_analysis_task'],
		)
		return head_to_head_analysis_task
	
	@task
	def current_season_performance_task(self) -> Task:

		current_season_performance_task = Task(
			config = self.tasks_config['current_season_performance_task'],
		)
		return current_season_performance_task
	
	@task
	def coaching_strategy_task(self) -> Task:

		coaching_strategy_task = Task(
			config = self.tasks_config['coaching_strategy_task'],
		)
		return coaching_strategy_task
	
	@task
	def environmental_impact_task(self) -> Task:

		environmental_impact_task = Task(
			config = self.tasks_config['environmental_impact_task'],
		)
		return environmental_impact_task

	@task
	def performance_summary_task(self) -> Task:

		performance_summary_task = Task(
			config = self.tasks_config['performance_summary_task'],
		)
		return performance_summary_task	
	@task
	def consensus_summary_task(self) -> Task:

		consensus_summary = Task(
			config=self.tasks_config['consensus_summary_task'],
		)

		return consensus_summary
	

	@crew
	def crew(self) -> Crew:
		"""Creates the NflPredicitonAssistant crew"""

		crew_workflow = Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			# process=Process.hierarchical,
			process=Process.sequential,
			verbose=True,
			memory=True,
			knowledge_sources=[self.json_source, self.csv_source],
			manager_agent=self.manager,
		)

		return crew_workflow
