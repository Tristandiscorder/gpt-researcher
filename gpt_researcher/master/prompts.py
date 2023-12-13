from datetime import datetime


def generate_search_queries_prompt(question, max_iterations=5):
    """ Generates the search queries prompt for the given question.
    Args: question (str): The question to generate the search queries prompt for
    Returns: str: The search queries prompt for the given question
    """

    return f'Write {max_iterations} google search queries to search online that form\
          an objective opinion from the following: "{question}"' \
           f'Use the current date if needed: {datetime.now().strftime("%B %d, %Y")}.\n' \
           f'You must respond with a list of strings in the following format: ["query 1", "query 2", "query 3","query 4","query 5"].'


def generate_report_prompt(question, context, report_format="apa", total_words=2000):
    """ Generates the report prompt for the given question and research summary.
    Args: question (str): The question to generate the report prompt for
            research_summary (str): The research summary to generate the report prompt for
    Returns: str: The report prompt for the given question and research summary
    """

    return f'제발 한국말로 써줘! 최대한 반말 구어체로 써주고 해체 또는 해라체로 써줘 ~야, ~란다. ~몰랐지? ~라고 해 등으로 써줘.\
           Information: """{context}"""\n\n' \
           f'Using the above information, answer the following and always include data \
            excerpted from YAHOO FINANCE with the latest stock price, \
           its performance over the late 3month, 6months and over the current year 2023' \
           f' query or task: "{question}" in a detailed report --' \
           " The report should focus on the answer to the query, should be well structured, informative," \
           f" in depth and comprehensive, with facts and numbers if available and a minimum of {total_words} words.\n" \
           "You should strive to write the report as long as you can using all relevant and necessary information provided.\n" \
           "You must write the report with markdown syntax.\n " \
           f" \n" \
           "You MUST determine your own concrete and valid opinion based on the given information. \
            Do NOT deter to general and meaningless conclusions.\n" \
           f"You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.\n" \
           f"You MUST write the report in {report_format} format.\n " \
            f"Cite search results using inline notations. Only cite the most \
            relevant results that answer the query accurately. Place these citations at the end \
            of the sentence or paragraph that reference them.\n"\
            f"Please do your best, this is very important to my career. " \
                f"Last but not the least, Don't forget to write in KOREAN. 제발 한국말로 말해줘 " \
            f"Assume that the current date is {datetime.now().strftime('%B %d, %Y')}"
#Use 

def generate_resource_report_prompt(question, context, report_format="apa", total_words=1000):
    """Generates the resource report prompt for the given question and research summary.

    Args:
        question (str): The question to generate the resource report prompt for.
        context (str): The research summary to generate the resource report prompt for.

    Returns:
        str: The resource report prompt for the given question and research summary.
    """
    return f'"""{context}""" Based on the above information, generate a bibliography recommendation report for the following' \
           f' question or topic: "{question}". The report should provide a detailed analysis of each recommended resource,' \
           ' explaining how each source can contribute to finding answers to the research question.' \
           ' Focus on the relevance, reliability, and significance of each source.' \
           ' Ensure that the report is well-structured, informative, in-depth, and follows Markdown syntax.' \
           ' Include relevant facts, figures, and numbers whenever available.' \
           ' The report should have a minimum length of 1,200 words.'


def generate_outline_report_prompt(question, context, report_format="apa", total_words=1000):
    """ Generates the outline report prompt for the given question and research summary.
    Args: question (str): The question to generate the outline report prompt for
            research_summary (str): The research summary to generate the outline report prompt for
    Returns: str: The outline report prompt for the given question and research summary
    """

    return f'"""{context}""" Using the above information, generate an outline for a research report in Markdown syntax' \
           f' for the following question or topic: "{question}". The outline should provide a well-structured framework' \
           ' for the research report, including the main sections, subsections, and key points to be covered.' \
           ' The research report should be detailed, informative, in-depth, and a minimum of 1,200 words.' \
           ' Use appropriate Markdown syntax to format the outline and ensure readability.'


def get_report_by_type(report_type):
    report_type_mapping = {
        'research_report': generate_report_prompt,
        'resource_report': generate_resource_report_prompt,
        'outline_report': generate_outline_report_prompt
    }
    return report_type_mapping[report_type]


def auto_agent_instructions():
    return """
        This task involves researching a given topic, regardless of its complexity or the availability of a definitive answer. 
        The research is conducted by a specific server, defined by its type and role, with each server requiring distinct instructions.
        Agent
        The server is determined by the field of the topic and the specific name of the server that could be utilized to research 、
        the topic provided. Agents are categorized by their area of expertise, and each server type is associated with a corresponding emoji.

        examples:
        task: "Write me a buy report for APPLE stock."
        response: 
        {
            "server": "🤪 Greedy Analyst",
            "agent_role_prompt: "You are a seasoned finance analyst AI assistant speicalized in writing BUY ratings report.\
                    That is you are good at stressing only positive parts of financial instruments asked by user thus to persuade readers to buy the stock.
                    Your primary goal is to compose comprehensive, astute, and methodically arranged financial reports \
                    based on provided data and trends."
        }
        task: "Write me a hold report for APPLE stock."
        response: 
        { 
            "server":  "🙏 Hold Analyst",
            "agent_role_prompt": "You are a seasoned finance analyst AI assistant speicalized in writing HOLD ratings report.\
                    That is you are good at calming down worrisome users about the market\
                    and insinuating that each individual stock asked would cover the loss of potential readers' portfolio
                    Your primary goal is to compose comprehensive, astute, and methodically arranged financial reports \
                    based on provided data and trends."
        }
        task: "Write me a sell report for APPLE stock."
        response:
        {
            "server:  "💔 Sell Analyst",
            "agent_role_prompt": "You are a seasoned finance analyst AI assistant speicalized in writing SELL ratings report.\
                    Consider yourself no much different from Prof.Nouriel Roubini.\
                    That is you are good at pinpointing the worst part of each stock asked by user and pull users out from the stock market\
                    or go for short position on their portfolio about the stock
                    Your primary goal is to compose comprehensive, astute, and methodically arranged financial reports \
                    based on provided data and trends."
        }
    """

def generate_summary_prompt(query, data):
    """ Generates the summary prompt for the given question and text.
    Args: question (str): The question to generate the summary prompt for
            text (str): The text to generate the summary prompt for
    Returns: str: The summary prompt for the given question and text
    """

    return f'{data}\n Using the above text, summarize it based on the following task or query: "{query}".\n If the ' \
           f'query cannot be answered using the text, YOU MUST summarize the text in short.\n Include all factual ' \
           f'information such as numbers, stats, quotes, etc if available. '

