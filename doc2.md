3. Créer DocAITest/DocAITest.csproj
```

<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="DocumentFormat.OpenXml" Version="3.0.2" />
    <Compile Include="..\DocumentAIProcessor.cs" Link="DocumentAIProcessor.cs" />
  </ItemGroup>
</Project>
```
4. Créer DocAITest/Program.cs

```
using DocAI;

var inputFile = args.Length > 0 ? args[0] : @"..\Docs\test.docx";
var outputFile = args.Length > 1 ? args[1] : @"..\Docs\test_output.docx";
var instruction = args.Length > 2 ? args[2] : "corrige les erreurs";
var mode = args.Length > 3 && args[3] == "s" ? ProcessingMode.Suggestion : ProcessingMode.Modification;

await Test.RunAsync(inputFile, outputFile, instruction, mode);
```
5. Installer et lancer

cd DocAITest
dotnet restore
dotnet run -- "..\Docs\votre_fichier.docx" "..\Docs\output.docx" "corrige" m






```
// DocumentAIProcessor.cs - Single-file for Sinequa plugin
// NuGet: DocumentFormat.OpenXml

using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;
using WRun = DocumentFormat.OpenXml.Wordprocessing.Run;
using WText = DocumentFormat.OpenXml.Wordprocessing.Text;
using WParagraph = DocumentFormat.OpenXml.Wordprocessing.Paragraph;
using WComments = DocumentFormat.OpenXml.Wordprocessing.Comments;
using WComment = DocumentFormat.OpenXml.Wordprocessing.Comment;

namespace DocAI;

#region Models

public enum ProcessingMode { Modification, Suggestion }

public record TextBlock(int Id, string Text);
public record ProcessedBlock(int Id, string Original, string Modified, string? Comment = null);

/// <summary>
/// Format JSON envoyé au LLM
/// </summary>
public record LlmRequest(string Instruction, List<TextBlock> Blocks);

#endregion

#region Interface - À implémenter avec Sinequa LLM Connector

/// <summary>
/// Interface LLM - reçoit JSON, retourne JSON
///
/// Entrée (LlmRequest en JSON):
/// {
///   "instruction": "corrige les erreurs",
///   "blocks": [
///     { "id": 0, "text": "Le projet Apollo..." },
///     { "id": 1, "text": "Autre paragraphe..." }
///   ]
/// }
///
/// Sortie (List&lt;ProcessedBlock&gt; en JSON):
/// [
///   { "id": 0, "original": "...", "modified": "...", "comment": "raison" },
///   { "id": 1, "original": "...", "modified": "...", "comment": null }
/// ]
/// </summary>
public interface ITextProcessor
{
    Task<List<ProcessedBlock>> ProcessAsync(LlmRequest request);
}

#endregion

#region Main Processor

public class DocumentAIProcessor(ITextProcessor processor, int batchSize = 20)
{
    public async Task<byte[]> ProcessAsync(byte[] data, string instruction, ProcessingMode mode = ProcessingMode.Modification)
    {
        // Clone: on travaille sur une copie, l'original reste intact
        var clone = new byte[data.Length];
        Array.Copy(data, clone, data.Length);

        // Extract text blocks
        var blocks = ExtractBlocks(clone);
        if (blocks.Count == 0) return clone;

        // Process via LLM par lots
        var processed = await ProcessInBatchesAsync(blocks, instruction);
        var changes = processed.Where(p => p.Modified != p.Original).ToList();
        if (changes.Count == 0) return clone;

        // Apply to cloned document
        using var ms = new MemoryStream();
        ms.Write(clone, 0, clone.Length);
        ms.Position = 0;

        using (var doc = WordprocessingDocument.Open(ms, true))
        {
            var body = doc.MainDocumentPart!.Document.Body!;
            var paragraphs = body.Descendants<WParagraph>()
                .Where(p => !string.IsNullOrWhiteSpace(GetText(p)))
                .ToList();

            var lookup = changes.ToDictionary(c => c.Id);

            for (int i = 0; i < paragraphs.Count && i < blocks.Count; i++)
            {
                if (!lookup.TryGetValue(i, out var change)) continue;

                if (mode == ProcessingMode.Modification)
                    ReplaceParagraph(paragraphs[i], change.Modified);
                else
                    AddComment(doc, paragraphs[i], change.Modified, change.Comment);
            }

            doc.MainDocumentPart.Document.Save();
        }

        return ms.ToArray();
    }

    private async Task<List<ProcessedBlock>> ProcessInBatchesAsync(List<TextBlock> blocks, string instruction)
    {
        var results = new List<ProcessedBlock>();

        foreach (var batch in blocks.Chunk(batchSize))
        {
            var request = new LlmRequest(instruction, batch.ToList());
            var response = await processor.ProcessAsync(request);
            results.AddRange(response);
        }

        return results;
    }

    private List<TextBlock> ExtractBlocks(byte[] data)
    {
        var blocks = new List<TextBlock>();
        using var ms = new MemoryStream(data);
        using var doc = WordprocessingDocument.Open(ms, false);

        var body = doc.MainDocumentPart?.Document.Body;
        if (body == null) return blocks;

        int id = 0;
        foreach (var p in body.Descendants<WParagraph>())
        {
            var text = GetText(p);
            if (!string.IsNullOrWhiteSpace(text))
                blocks.Add(new TextBlock(id++, text));
        }
        return blocks;
    }

    private static string GetText(WParagraph p) =>
        string.Join("", p.Descendants<WText>().Select(t => t.Text));

    private static void ReplaceParagraph(WParagraph p, string newText)
    {
        var runs = p.Descendants<WRun>().ToList();
        if (runs.Count == 0) return;

        var props = runs[0].RunProperties?.CloneNode(true);
        runs.ForEach(r => r.Remove());

        var newRun = new WRun();
        if (props != null) newRun.AppendChild(props);
        newRun.AppendChild(new WText(newText) { Space = SpaceProcessingModeValues.Preserve });
        p.AppendChild(newRun);
    }

    private static void AddComment(WordprocessingDocument doc, WParagraph p, string suggestion, string? reason)
    {
        var part = doc.MainDocumentPart!.WordprocessingCommentsPart
                   ?? doc.MainDocumentPart.AddNewPart<WordprocessingCommentsPart>();
        part.Comments ??= new WComments();

        var id = part.Comments.Count().ToString();
        var text = string.IsNullOrEmpty(reason)
            ? $"Suggestion: {suggestion}"
            : $"{reason}\n\nSuggestion: {suggestion}";

        var comment = new WComment { Id = id, Author = "DocAI" };
        comment.AppendChild(new WParagraph(new WRun(new WText(text))));
        part.Comments.AppendChild(comment);

        var first = p.Descendants<WRun>().FirstOrDefault();
        if (first != null)
        {
            p.InsertBefore(new CommentRangeStart { Id = id }, first);
            p.AppendChild(new CommentRangeEnd { Id = id });
            p.AppendChild(new WRun(new CommentReference { Id = id }));
        }
    }
}

#endregion

#region Mock LLM Processor (pour test)

public class MockLlmProcessor : ITextProcessor
{
    public Task<List<ProcessedBlock>> ProcessAsync(LlmRequest request)
    {
        Console.WriteLine($"  [LLM] Batch de {request.Blocks.Count} blocs, instruction: \"{request.Instruction}\"");

        var results = request.Blocks.Select(b =>
        {
            var modified = request.Instruction.ToLowerInvariant() switch
            {
                var i when i.Contains("majuscule") => b.Text.ToUpperInvariant(),
                var i when i.Contains("minuscule") => b.Text.ToLowerInvariant(),
                var i when i.Contains("corrige") => b.Text.Replace("Allan", "Alan"),
                _ => b.Text
            };

            if (modified != b.Text)
            {
                Console.WriteLine($"    [ID {b.Id}] Changement détecté:");
                Console.WriteLine($"      Avant: {b.Text.Substring(0, Math.Min(50, b.Text.Length))}...");
                Console.WriteLine($"      Après: {modified.Substring(0, Math.Min(50, modified.Length))}...");
            }

            return new ProcessedBlock(
                b.Id,
                b.Text,
                modified,
                modified != b.Text ? $"Appliqué: {request.Instruction}" : null
            );
        }).ToList();

        Console.WriteLine($"  [LLM] {results.Count(r => r.Modified != r.Original)} modifications sur {results.Count} blocs");

        return Task.FromResult(results);
    }
}

#endregion

#region Test

public static class Test
{
    public static async Task RunAsync(string inputFile, string outputFile, string instruction, ProcessingMode mode, int batchSize = 20)
    {
        if (!File.Exists(inputFile))
        {
            Console.WriteLine($"Fichier introuvable: {inputFile}");
            return;
        }

        Console.WriteLine($"Input:       {inputFile}");
        Console.WriteLine($"Output:      {outputFile}");
        Console.WriteLine($"Instruction: {instruction}");
        Console.WriteLine($"Mode:        {mode}");
        Console.WriteLine($"Batch size:  {batchSize}");
        Console.WriteLine();

        var processor = new DocumentAIProcessor(new MockLlmProcessor(), batchSize);
        var input = await File.ReadAllBytesAsync(inputFile);
        var output = await processor.ProcessAsync(input, instruction, mode);

        await File.WriteAllBytesAsync(outputFile, output);

        Console.WriteLine();
        Console.WriteLine($"Done! Original intact, modifications dans: {outputFile}");
    }
}

#endregion

```