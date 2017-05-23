from torch.nn import BCECriterion
from torch.autograd import Variable
from torch import FloatTensor

class GAN(object):
    def __init__(self, generator, discriminator, optimizers,
                 batch_size, latent_dim, dim):
        self.gen = generator
        self.disc = discriminator

        self.gen_optimizer = optimizers[0]
        self.disc_optimizer = optimizers[1]

        self.labels = Variable(FloatTensor(batch_size))
        self.input_size = (batch_size, latent_dim) + (1)*dim
        self.latent_vectors = Variable(FloatTensor(self.input_size))
        self.vars = [self.gen, self.disc, self.labels, self.noise]

    def step(self):
        return None

    def to_cuda(self):
        self.vars = [var.cuda() for var in self.vars]


class VanillaGAN(GAN):
    def __init__(self, generator, discriminator, batch_size, latent_dim, dim):
        super(VanillaGAN, self).__init__(generator, discriminator, batch_size,
                                         latent_dim, dim)
        self.criterion = BCECriterion()
        self.real_label = 1
        self.fake_label = 0

    def step(self, input_batch):
        #zero the gradients
        self.disc.zero_grad()
        err_D_x, D_x = self.discriminator_real_step(input_batch)
        err_D_G_z, D_G_z = self.discriminator_fake_step()

        expectation_D_x = D_x.data.mean()
        expectation_D_G_z = D_G_z.data.mean()

        err_D = err_D_x + err_D_G_z
        self.disc_optimizer.step()

        #zero the grads for the generator
        self.gen.zero_grad()
        err_D_G_z2, D_G_z2 = self.generator_step()
        self.gen_optimizer.step()

        expectation_D_G_z2 = D_G_z2.data.mean()

        return expectation_D_x, expectation_D_G_z, expectation_D_G_z2

    def discriminator_real_step(self, input_batch):
        self.labels.data.fill_(self.real_label)
        D_x = self.disc(input_batch)
        err_D_x = self.criterion(D_x, self.labels)
        err_D_x.backward()
        return err_D_x, D_x

    def discriminator_fake_step(self):
        self.labels.data.fill_(self.fake_label)
        G_z = self.generate_G_z()
        D_G_z = self.disc(G_z)
        err_D_G_z = self.criterion(D_G_z, self.labels)
        err_D_G_z.backward()
        return err_D_G_z, D_G_z

    def generate_G_z(self):
        self.latent_vectors.data.normal_(0, 1)
        G_z = self.gen(self.latent_vectors).detach()
        return G_z

    def generator_step(self):
        G_z = self.generate_G_z()
        self.labels.data.fill_(self.real_label)
        D_G_z = self.disc(G_z)
        err_D_G_z = self.criterion(D_G_z, self.labels)
        err_D_G_z.backward()

        return err_D_G_z, D_G_z