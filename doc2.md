import os
import httpx
import mistralai.workflows as workflows

API = "https://api.mistral.ai/v1"

@workflows.activity.define(name="get_triggering_user_email")   # appels HTTP = dans une activité
async def get_triggering_user_email(execution_id: str) -> str | None:
    async with httpx.AsyncClient(timeout=10) as c:
        # 1. Qui a lancé l'exécution ?
        r = await c.get(f"{API}/workflows/executions/{execution_id}",
                        headers={"Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}"})
        r.raise_for_status()
        user_id = r.json().get("user_id")
        if not user_id:
            return None  # ex. lancé par un schedule ou un service account

        # 2. Son email (clé admin requise)
        r = await c.get(f"{API}/admin/users/{user_id}",
                        headers={"Authorization": f"Bearer {os.environ['MISTRAL_ADMIN_API_KEY']}"})
        r.raise_for_status()
        return r.json().get("email")


@workflows.workflow.define(name="my_workflow")
class MyWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, params: MyParams) -> MyResult:
        email = await get_triggering_user_email(workflows.get_execution_id())
        ...