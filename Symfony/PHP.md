# Commandes Console PHP

---

1. **Accéder au container** :
   ```bash
   docker exec -ti projet_test /bin/bash
   ```

2. **Lister les routes** :
   ```bash
   php bin/console debug:router
   ```

3. **Lister les commandes** :
   ```bash
   php bin/console list
   ```

4. **Vider le cache** :
   ```bash
   php bin/console cache:clear
   php bin/console c:c
   ```

5. **Lancer les tests unitaires** :
   ```bash
   php bin/phpunit
   ```

6. **Exécuter une commande spécifique** :
   ```bash
   php bin/console app:LA_COMMAND
   ```

7. **Créer des éléments** :
   ```bash
   php bin/console make:[action]
   # Exemples : 
   php bin/console make:entity
   php bin/console make:form
   php bin/console make:auth
   php bin/console make:migration
   php bin/console make:factory
   php bin/console make:registration-form
   ```

8. **Exécuter les migrations** :
   ```bash
   php bin/console doctrine:migrations:migrate
   php bin/console doctrine:migrations:exec <version> --up
   php bin/console doctrine:migrations:exec <version> --down
   ```

9. **Charger les fixtures** :
   ```bash
   php bin/console doctrine:fixtures:load
   ```