from config_final_metrics import config
import os
import tensorflow as tf
from load_data import load_dSprites
from latent import UniformLatent, JointLatent
from network import Decoder, InfoGANDiscriminator, \
    CrDiscriminator, MetricRegresser
from infogan_cr import INFOGAN_CR
from metric import FactorVAEMetric, DSpritesInceptionScore, \
    DHSICMetric, \
    BetaVAEMetric, SAPMetric, FStatMetric, MIGMetric, DCIMetric
import pickle


def main():
    
    data, metric_data, latent_values, metadata = \
        load_dSprites("../data/dSprites")
    _, height, width, depth = data.shape

    latent_list = []

    for i in range(config["uniform_reg_dim"]):
        latent_list.append(UniformLatent(
            in_dim=1, out_dim=1, low=-1.0, high=1.0, q_std=1.0,
            apply_reg=True))
    if config["uniform_not_reg_dim"] > 0:
        latent_list.append(UniformLatent(
            in_dim=config["uniform_not_reg_dim"],
            out_dim=config["uniform_not_reg_dim"],
            low=-1.0, high=1.0, q_std=1.0,
            apply_reg=False))
    latent = JointLatent(latent_list=latent_list)

    decoder = Decoder(
        output_width=width, output_height=height, output_depth=depth)
    infoGANDiscriminator = \
        InfoGANDiscriminator(
            output_length=latent.reg_out_dim,
            q_l_dim=config["q_l_dim"])
    crDiscriminator = CrDiscriminator(output_length=latent.num_reg_latent)

    shape_network = MetricRegresser(
        output_length=3,
        scope_name="dSpritesSampleQualityMetric_shape")

    work_dir = './test/'
    checkpoint_dir = os.path.join(work_dir, "checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    sample_dir = os.path.join(work_dir, "sample")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    time_path = os.path.join(work_dir, "time.txt")
    metric_path = os.path.join(work_dir, "metric.csv")

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
        factorVAEMetric = FactorVAEMetric(metric_data, sess=sess)
        dSpritesInceptionScore = DSpritesInceptionScore(
            sess=sess,
            do_training=False,
            data=data,
            metadata=metadata,
            latent_values=latent_values,
            network_path="../metric_model/DSprites",
            shape_network=shape_network,
            sample_dir=sample_dir)
        dHSICMetric = DHSICMetric(
            sess=sess,
            data=data)
        metric_callbacks = [factorVAEMetric,
                            dSpritesInceptionScore,
                            dHSICMetric]
        gan = INFOGAN_CR(
            sess=sess,
            checkpoint_dir=checkpoint_dir,
            sample_dir=sample_dir,
            time_path=time_path,
            epoch=config["epoch"],
            batch_size=config["batch_size"],
            data=data,
            vis_freq=config["vis_freq"],
            vis_num_sample=config["vis_num_sample"],
            vis_num_rep=config["vis_num_rep"],
            latent=latent,
            decoder=decoder,
            infoGANDiscriminator=infoGANDiscriminator,
            crDiscriminator=crDiscriminator,
            gap_start=config["gap_start"],
            gap_decrease_times=config["gap_decrease_times"],
            gap_decrease=config["gap_decrease"],
            gap_decrease_batch=config["gap_decrease_batch"],
            cr_coe_start=config["cr_coe_start"],
            cr_coe_increase_times=config["cr_coe_increase_times"],
            cr_coe_increase=config["cr_coe_increase"],
            cr_coe_increase_batch=config["cr_coe_increase_batch"],
            info_coe_de=config["info_coe_de"],
            info_coe_infod=config["info_coe_infod"],
            metric_callbacks=metric_callbacks,
            metric_freq=config["metric_freq"],
            metric_path=metric_path,
            output_reverse=config["output_reverse"],
            de_lr=config["de_lr"],
            infod_lr=config["infod_lr"],
            crd_lr=config["crd_lr"],
            summary_freq=config["summary_freq"])
        gan.build()
        gan.load()

        results = {}

        factorVAEMetric_f = FactorVAEMetric(metric_data, sess=sess)
        factorVAEMetric_f.set_model(gan)
        results["FactorVAE"] = factorVAEMetric_f.evaluate(-1, -1, -1)

        betaVAEMetric_f = BetaVAEMetric(metric_data, sess=sess)
        betaVAEMetric_f.set_model(gan)
        results["betaVAE"] = betaVAEMetric_f.evaluate(-1, -1, -1)
        
        sapMetric_f = SAPMetric(metric_data, sess=sess)
        sapMetric_f.set_model(gan)
        results["SAP"] = sapMetric_f.evaluate(-1, -1, -1)

        fStatMetric_f = FStatMetric(metric_data, sess=sess)
        fStatMetric_f.set_model(gan)
        results["FStat"] = fStatMetric_f.evaluate(-1, -1, -1)

        migMetric_f = MIGMetric(metric_data, sess=sess)
        migMetric_f.set_model(gan)
        results["MIG"] = migMetric_f.evaluate(-1, -1, -1)

        for regressor in ["Lasso", "LassoCV", "RandomForest", "RandomForestIBGAN", "RandomForestCV"]:
            dciVAEMetric_f = DCIMetric(metric_data, sess=sess, regressor=regressor)
            dciVAEMetric_f.set_model(gan)
            results["DCI_{}".format(regressor)] = dciVAEMetric_f.evaluate(-1, -1, -1)

        with open(os.path.join(work_dir, "final_metrics.pkl"), "wb") as f:
            pickle.dump(results, f)

if __name__ == '__main__':
    main()
