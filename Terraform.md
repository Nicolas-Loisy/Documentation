# Terraform

- [Formation GCP](https://www.cloudskillsboost.google/catalog_lab/4981)
- [Terraform Cheat Sheet](https://cheat-sheets.nicwortel.nl/terraform-cheat-sheet.pdf)
- [Terraform Cheat Sheet Repo](https://github.com/scraly/terraform-cheat-sheet/blob/master/terraform-cheat-sheet.pdf)

Terraform est un outil d'infrastructure en tant que code (IaC) qui permet de définir, déployer et gérer des infrastructures via des fichiers de configuration. Il est compatible avec de nombreux fournisseurs cloud (AWS, Azure, GCP) et services locaux.

---

### Concepts clés

- **Providers** : Connectent Terraform à une plateforme (ex. AWS, Azure).  
- **Ressources** : Objets gérés (ex. VM, bases de données).  
- **Variables** : Paramètres pour rendre les configurations dynamiques.  
- **State** : Fichier (`terraform.tfstate`) qui suit l’état des ressources.

---

### Commandes essentielles

1. **`terraform init`** : Initialise le projet.  
2. **`terraform plan`** : Prévisualise les modifications.  
3. **`terraform apply`** : Applique les changements.  
4. **`terraform destroy`** : Supprime les ressources.

---

### Exemple minimal (AWS)

```hcl
provider "aws" {
  region = "us-east-1"
}

resource "aws_instance" "example" {
  ami           = "ami-xxxxxxxxxxxxxxx"
  instance_type = "t2.micro"
}

output "instance_id" {
  value = aws_instance.example.id
}
```

Déploiement :  
```bash
terraform init
terraform plan
terraform apply
```
