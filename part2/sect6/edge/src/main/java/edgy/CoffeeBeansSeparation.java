package edgy;

import static marvin.MarvinPluginCollection.floodfillSegmentation;
import static marvin.MarvinPluginCollection.thresholding;

import java.awt.Color;
import java.util.ArrayList;
import java.util.List;

import marvin.color.MarvinColorModelConverter;
import marvin.image.MarvinImage;
import marvin.image.MarvinSegment;
import marvin.io.MarvinImageIO;
import marvin.math.MarvinMath;
import marvin.plugin.MarvinImagePlugin;
import marvin.util.MarvinPluginLoader;

public class CoffeeBeansSeparation {

    private MarvinImagePlugin erosion = MarvinPluginLoader.loadImagePlugin("org.marvinproject.image.morphological.erosion.jar");

    public CoffeeBeansSeparation(){

        MarvinImage image = MarvinImageIO.loadImage("./coffee_input.png");
        MarvinImage result = image.clone();

        thresholding(image, 30);
        MarvinImageIO.saveImage(image, "./res/coffee_threshold.png");


        List<MarvinSegment> listSegments = new ArrayList<MarvinSegment>();
        List<MarvinSegment> listSegmentsTmp = new ArrayList<MarvinSegment>();
        MarvinImage binImage = MarvinColorModelConverter.rgbToBinary(image, 127);
        erosion.setAttribute("matrix", MarvinMath.getTrueMatrix(8, 8));
        erosion.process(binImage.clone(), binImage);
        MarvinImageIO.saveImage(binImage, "./res/coffee_bin_8.png");
        
        MarvinImage binImageRGB = MarvinColorModelConverter.binaryToRgb(binImage);
        MarvinSegment[] segments =  floodfillSegmentation(binImageRGB);
        for(MarvinSegment s:segments){
            if(s.area < 300){   
                listSegments.add(s);
            }
        }
        showSegments(listSegments, binImageRGB);
        MarvinImageIO.saveImage(binImageRGB, "./res/coffee_center_8.png");

        // 5. Segment using erosion and floodfill (kernel size == 18)
        listSegments = new ArrayList<MarvinSegment>();
        binImage = MarvinColorModelConverter.rgbToBinary(image, 127);

        erosion.setAttribute("matrix", MarvinMath.getTrueMatrix(18, 18));
        erosion.process(binImage.clone(), binImage);

        MarvinImageIO.saveImage(binImage, "./res/coffee_bin_8.png");
        binImageRGB = MarvinColorModelConverter.binaryToRgb(binImage);
        segments =  floodfillSegmentation(binImageRGB);

        for(MarvinSegment s:segments){
            listSegments.add(s);
            listSegmentsTmp.add(s);
        }

        showSegments(listSegmentsTmp, binImageRGB);
        MarvinImageIO.saveImage(binImageRGB, "./res/coffee_center_18.png");

        // 6. Remove segments that are too near.
        MarvinSegment.segmentMinDistance(listSegments, 10);

        // 7. Show Result
        showSegments(listSegments, result);
        MarvinImageIO.saveImage(result, "./res/coffee_result.png");
    }

    private void showSegments(List<MarvinSegment> segments, MarvinImage image){
        for(MarvinSegment s:segments){
            image.fillRect((s.x1+s.x2)/2, (s.y1+s.y2)/2, 5, 5, Color.red);
        }
    }

    public static void main(String[] args) {
        new CoffeeBeansSeparation();
    }
}