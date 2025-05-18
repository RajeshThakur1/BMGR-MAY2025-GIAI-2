import json
BASE_DIR = "/Users/rajeshthakur/Desktop/rajesh/Archive/teaching/agentic_ai/BMGR-MAY2025-GIAI-2/"
def read_user():
    with open(f"{BASE_DIR}data/users.json") as stream:
        users = json.load(stream)
        return users


def read_questions(position: int):
    with open(f"{BASE_DIR}data/questions.json") as stream:
        questions = json.load(stream)

    for question in questions:
        if question['position'] == position:
            res = question.get('question')
            print(res)
            return res






if __name__ == "__main__":
    print(read_user())
    read_questions(1)





