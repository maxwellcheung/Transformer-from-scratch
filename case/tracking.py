def init_swanlab(args, output_dir, train_dataset, valid_dataset, model):
    if not args.swanlab:
        return None

    try:
        import swanlab
    except ImportError as error:
        raise RuntimeError(
            'SwanLab tracking requested but swanlab is not installed; '
            'run `python -m pip install -r case/requirements.txt`'
        ) from error

    config = dict(vars(args))
    config.update({
        'train_samples_after_filtering': len(train_dataset),
        'valid_samples_after_filtering': len(valid_dataset),
        'train_samples_skipped_too_long': (
            train_dataset.skipped_too_long
        ),
        'valid_samples_skipped_too_long': (
            valid_dataset.skipped_too_long
        ),
        'model_parameters': sum(
            parameter.numel() for parameter in model.parameters()
        ),
    })

    init_args = {
        'project': args.swanlab_project,
        'experiment_name': (
            args.swanlab_experiment_name or output_dir.name
        ),
        'description': (
            'Chinese-to-English Transformer trained from scratch'
        ),
        'config': config,
        'logdir': str(output_dir / 'swanlog'),
        'mode': args.swanlab_mode,
        'on_online_error': 'offline',
    }
    if args.swanlab_workspace:
        init_args['workspace'] = args.swanlab_workspace

    return swanlab.init(**init_args)
