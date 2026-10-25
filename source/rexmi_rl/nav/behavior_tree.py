"""Small reactive tree: decisions return RUNNING without restarting an action."""
from enum import Enum


class Status(Enum):
    FAILURE = 0
    RUNNING = 1
    SUCCESS = 2


class Action:
    def __init__(self, name, callback):
        self.name, self.callback = name, callback

    def tick(self, context):
        result = self.callback(context)
        if result is not Status.FAILURE:
            context.tree_branch = self.name
        return result


class ReactiveSelector:
    """Re-evaluate priority each tick; the executor owns persistent action state."""
    def __init__(self, *children):
        self.children = children

    def tick(self, context):
        for child in self.children:
            result = child.tick(context)
            if result is not Status.FAILURE:
                return result
        return Status.FAILURE
