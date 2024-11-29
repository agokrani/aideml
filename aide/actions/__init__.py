from aide.actions.action import *

action_name_to_action = {
    "Draft": Draft,
    "Improve": Improve,
    "Debug": Debug,
    "Finish": Finish,
    "SubmitReview": SubmitReview
}

def get_action(action_name: str) -> BaseModel:
    if action_name in action_name_to_action.keys(): 
        return action_name_to_action[action_name]
    else:
        return None
