import importlib
import sys
import types
import unittest


def _install_stubs():
    ollama = types.ModuleType("ollama")
    pandas = types.ModuleType("pandas")

    agent1_main = types.ModuleType("agent1.main")
    agent1_main.MODEL_NAME = "test-model"
    agent1_main.conversation_context = lambda history=None: "[]"

    agent1_pkg = types.ModuleType("agent1")
    agent1_pkg.main = agent1_main

    agent2_main = types.ModuleType("agent2.main2")
    agent2_pkg = types.ModuleType("agent2")
    agent2_pkg.main2 = agent2_main

    agent3_main = types.ModuleType("agent3.agent3")
    agent3_pkg = types.ModuleType("agent3")
    agent3_pkg.agent3 = agent3_main

    sys.modules.setdefault("ollama", ollama)
    sys.modules.setdefault("pandas", pandas)
    sys.modules.setdefault("agent1", agent1_pkg)
    sys.modules.setdefault("agent1.main", agent1_main)
    sys.modules.setdefault("agent2", agent2_pkg)
    sys.modules.setdefault("agent2.main2", agent2_main)
    sys.modules.setdefault("agent3", agent3_pkg)
    sys.modules.setdefault("agent3.agent3", agent3_main)


class WeatherHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_stubs()
        cls.orchestrator = importlib.import_module("orchestrator")

    def test_weather_question_uses_city_rows(self):
        prior_answers = [{
            "agent_name": "E-commerce orders agent",
            "result_records": [
                {"city": "Munich", "country": "Germany", "customers": 20385},
                {"city": "Brussels", "country": "Belgium", "customers": 20232},
                {"city": "Ottawa", "country": "Canada", "customers": 20211},
            ],
        }]

        question = self.orchestrator._city_weather_question(
            "Find the top 3 customer cities and give me the current temperature in each city.",
            "Get the current temperature.",
            prior_answers,
        )

        self.assertIn("Munich, Germany", question)
        self.assertIn("Brussels, Belgium", question)
        self.assertIn("Ottawa, Canada", question)

    def test_weather_question_uses_country_capitals(self):
        prior_answers = [{
            "agent_name": "FIFA players agent",
            "result_records": [
                {"Nationality": "England", "players": 1657},
                {"Nationality": "Germany", "players": 1195},
                {"Nationality": "Spain", "players": 1071},
                {"Nationality": "Argentina", "players": 936},
                {"Nationality": "France", "players": 911},
            ],
        }]

        question = self.orchestrator._city_weather_question(
            "Find the top five countries by number of players and give me the current temperature in each country’s capital.",
            "Get current temperatures.",
            prior_answers,
        )

        self.assertIn("London, United Kingdom", question)
        self.assertIn("Berlin, Germany", question)
        self.assertIn("Madrid, Spain", question)
        self.assertIn("Buenos Aires, Argentina", question)
        self.assertIn("Paris, France", question)

    def test_weather_question_adds_mentioned_comparison_country(self):
        prior_answers = [{
            "agent_name": "FIFA players agent",
            "result_records": [
                {"Nationality": "Spain", "players_above_85": 10},
            ],
        }]

        question = self.orchestrator._city_weather_question(
            "Which country produces the most players rated above 85, and is it currently warmer there than Egypt?",
            "Compare the weather.",
            prior_answers,
        )

        self.assertIn("Madrid, Spain", question)
        self.assertIn("Cairo, Egypt", question)


if __name__ == "__main__":
    unittest.main()
