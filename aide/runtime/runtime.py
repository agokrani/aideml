import abc


class Runtime(abc.ABC):

    @abc.abstractmethod
    async def cleanup_session(self):
        pass

    @abc.abstractmethod
    async def run(self):
        pass
