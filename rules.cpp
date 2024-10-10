#include "rules.h"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif

#include "stb_image.h"

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION  // Define this in exactly one source file
#endif

#include "stb_image_write.h"

void Rules::Rule::writeImage(Grid &grid)
{
    if(patternImagesRoot == ""){
        return;
    }

    if(patterns.size() == 0){
        for (int i = 0; i < M; i++)
        {
            int width, height, channels;
            auto fileName = fs::path(std::to_string(i) + std::string(".png"));
            unsigned char* img = stbi_load((patternImagesRoot / fileName).c_str(), &width, &height, &channels, 0);
            if (img == nullptr) {
                std::cerr << "Failed to load image!\n" ;
                return;
            }

            if(i == 0){
                patternShape = ImageShape(width, height, channels);
                //std::cout << height << " " << width << " " << channels << "\n";
            }
            else{
                if(ImageShape(width, height, channels) != patternShape){
                    std::cerr << "Shape not match.\n"<< height << " " << width << " " << channels << "\n";
                    
                    return;
                }
            }
            patterns.push_back(img);
        }
    }

    assert(patterns.size() == M);

    std::cout << "Writing image...\n";
    int H = grid.size();
    int W = grid[0].size();

    int pH = patternShape.height;
    int pW = patternShape.width;

    int height = H * pH;
    int width = W * pW;

    int channels = patternShape.channels;  // RGB

    // Allocate memory for the image (RGB: 3 channels)
    std::shared_ptr<unsigned char[]> image(new unsigned char[width * height * channels](), std::default_delete<unsigned char[]>());

    // Fill the array with some pattern
    for (size_t h = 0; h < H; h++)
    {
        for (size_t w = 0; w < W; w++)
        {
            // for each cell
            Superposition state = grid[h][w];
            if(state.size() != 1){
                continue;
            }
            int patternId = *state.begin();
            auto pattern = patterns[patternId];
            auto index_of_first_pixel = (h * pH) * (width) + (w * pW);
            for (size_t i = 0; i < pH; i++)
            {
                for (size_t o = 0; o < pW; o++)
                {
                    // copy pattern
                    auto output_index = (index_of_first_pixel + (i * width) + o) * channels;
                    auto pattern_index = ((i * pW) + o) * channels;
                    for (size_t c = 0; c < channels; c++)
                    {
                        image[output_index + c] = pattern[pattern_index + c];  // Red
                    }
                }
            }
            
        }
    }
    

    // Save the image as a PNG
    if (stbi_write_png("output_image.png", width, height, channels, image.get(), width * channels)) {
        std::cout << "Image saved successfully!" << std::endl;
    } else {
        std::cerr << "Failed to save image!" << std::endl;
    }
}

Rules::Rule::~Rule()
{
    for(auto img: patterns){
        stbi_image_free(img);
    }
}
