name: Model Training CICD
permissions:
  id-token: write
  contents: write 
  
on:
  - push

jobs:
  run:
    runs-on: 
      - ubuntu-latest
    
    steps:
      - uses: iterative/setup-cml@v1
      - uses: actions/checkout@v3
      - name: Train model
        env:
          REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          pip install -r requirements.txt
          python train_model.py

          echo "RF and LR Model Score" >report.md
          cat scores.txt >> report.md
          
          echo "Confusion Matrix & Feature Importance">report1.md
          echo '|[](./ConfusionMatrix.png "ConfusionMatrix")' >> report1.md
          echo '|[](./FeatureImportance.png "FeatureImportance")' >> report1.md
     
          cat report.md report1.md >> combined_file.md
          cml comment create combined_file.md
