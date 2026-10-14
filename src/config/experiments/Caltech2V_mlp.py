from config.defaults import Experiment, DDC, Fusion, MLP, Loss, Dataset, THCMC, Optimizer



caltech2v = Experiment(
    dataset_config=Dataset(name="Caltech-2V"),
    model_config=THCMC(
        backbone_configs=(
            MLP(input_size=(40,)),
            MLP(input_size=(254,)),
        ),
        fusion_config=Fusion(method="original_weighted", n_views=2),
        projector_config=None,
        cm_config=DDC(n_clusters=7),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3|contrast|GraphCL|Contrastive_label_level",
            delta=20.0
        ),
        optimizer_config=Optimizer(scheduler_step_size=50, scheduler_gamma=0.1)
    ),

)

