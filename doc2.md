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
