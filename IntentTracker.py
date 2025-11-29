from collections import deque, Counter

class IntentTracker:
    def __init__(self, max_history: int):
        # 每个用户最多保留多少条历史意图
        self.max_history = max_history
        # user_id -> deque of intents
        self.user_history = {}

    def add(self, user_id: str, intent: str) -> None:
        """
        记录用户的一条新意图
        """
        if user_id not in self.user_history:
            self.user_history[user_id] = deque()
        q = self.user_history[user_id]
        if len(q) == self.max_history:
            q.popleft()
        q.append(intent)

    def get_top_k(self, user_id: str, k: int):
        """
        返回该用户最近 max_history 条意图中，出现频次最高的 k 个
        """
        if user_id not in self.user_history:
            return []
        q = self.user_history[user_id]
        cnt = Counter(q)
        # most_common(k) 返回 [(intent, freq), ...]
        return [intent for intent, _ in cnt.most_common(k)]
