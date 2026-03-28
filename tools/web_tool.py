from ddgs import DDGS
from langchain_core.tools import tool

@tool
def duckduckgo_search(query: str) -> str:
    try:
        # 获取前5条搜索结果
        summary = DDGS().text(query, max_results=5)
        return summary
    except Exception as e:
        return f"搜索出错: {e}"



__all__ = ["duckduckgo_search"]