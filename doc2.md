```
#nullable enable

using Assistant.Agentic.Workflow;
using Assistant.Agentic.Workflow.Executors;
using Microsoft.Agents.AI.Workflows;
using System;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace Sinequa.Plugin
{
    /// <summary>
    /// Example of a workflow plugin executor.
    /// It can read variables from the workflow, inspect the workflow state, and write a result for the next executor.
    /// </summary>
    public class MyDebugExecutorPlugin : AgentWorkflowExecutorPlugin
    {
        public MyDebugExecutorPlugin(string id) : base(id)
        {
        }

        public override async ValueTask ExecuteAsync(ExecutorPluginInput input, ExecutorPluginOutput output, IWorkflowContext workflowContext, CancellationToken cancellationToken = default)
        {
            Console.WriteLine($"[MyDebugExecutorPlugin] Start - pluginId={Id}, pluginName={PluginName}");

            // Example 1: read data already stored in the workflow by upstream executors
            var userQuery = await input.GetValueAsync<string>("userQuery").ConfigureAwait(false);
            var retryCount = await input.GetValueAsync<int?>("retryCount").ConfigureAwait(false);
            var payload = await input.GetValueAsync<JsonElement>("payload").ConfigureAwait(false);

            Console.WriteLine($"userQuery={userQuery ?? "<null>"}");
            Console.WriteLine($"retryCount={retryCount?.ToString() ?? "<null>"}");
            Console.WriteLine($"payload={payload.ToString()}");

            DebugDisplayMessage.AddDisplayItem("userQuery", userQuery ?? "<null>");
            DebugDisplayMessage.AddDisplayItem("retryCount", retryCount?.ToString() ?? "<null>");
            DebugDisplayMessage.AddDisplayItem("payload", payload.ValueKind != JsonValueKind.Undefined ? payload.ToString() : "<null>");

            // Example 2: inspect workflow state keys
            var workflowKeys = await workflowContext.ReadStateKeysAsync(scopeName: "_workflowScope", cancellationToken: cancellationToken).ConfigureAwait(false);
            Console.WriteLine("Workflow keys:");
            foreach (var key in workflowKeys)
            {
                Console.WriteLine($" - {key}");
            }

            DebugDisplayMessage.AddDisplayItem("Workflow keys", string.Join(", ", workflowKeys));

            // Example 3: output a value for next executor(s)
            await output.SetValueAsync("debugPluginResult", new
            {
                plugin = PluginName ?? nameof(MyDebugExecutorPlugin),
                userQuery,
                retryCount,
                keys = workflowKeys
            }).ConfigureAwait(false);

            Console.WriteLine($"[MyDebugExecutorPlugin] End - result stored under 'debugPluginResult'");
        }
    }
}

```