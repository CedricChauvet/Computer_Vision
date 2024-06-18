
from keras.preprocessing.image import ImageDataGenerator
prepa = __import__("02_preparation_des_donnees").prepa

x_train,y_train_ohe, x_val, y_val_ohe = prepa()
BATCH_SIZE = 32
def augmentation(x_train,y_train_ohe, x_val, y_val_ohe ):
    # Create train generator.
    train_datagen = ImageDataGenerator(rescale=1./255, 
                                    rotation_range=30, 
                                    width_shift_range=0.2,
                                    height_shift_range=0.2, 
                                    horizontal_flip = 'true')
    train_generator = train_datagen.flow(x_train, y_train_ohe, shuffle=False, 
                                        batch_size=BATCH_SIZE, seed=1)
                                        
    # Create validation generator
    val_datagen = ImageDataGenerator(rescale = 1./255)
    val_generator = train_datagen.flow(x_val, y_val_ohe, shuffle=False, 
                                    batch_size=BATCH_SIZE, seed=1)  

    return train_generator, val_generator












