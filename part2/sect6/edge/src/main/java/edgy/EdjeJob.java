package edgy;

import static marvin.MarvinPluginCollection.floodfillSegmentation;
import static marvin.MarvinPluginCollection.thresholding;

import java.awt.Color;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.NLineInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import marvin.color.MarvinColorModelConverter;
import marvin.image.MarvinImage;
import marvin.image.MarvinSegment;
import marvin.io.MarvinImageIO;
import marvin.math.MarvinMath;
import marvin.plugin.MarvinImagePlugin;
import marvin.util.MarvinPluginLoader;

public class EdjeJob extends Configured implements Tool {

	public static void main(String[] args) throws Exception {
		EdjeJob e = new EdjeJob();
		Configuration c = new Configuration();
		ToolRunner.run(c, e, new String[] {});
	}

	public static class TokenizerMapper extends Mapper<LongWritable, Text, Text, Text> {
		FileSystem fs;
		MarvinImagePlugin erosion;
		
	    @Override
		protected void setup(Mapper<LongWritable, Text, Text, Text>.Context context)
				throws IOException, InterruptedException {
			super.setup(context);
			fs = FileSystem.get(context.getConfiguration());
			erosion = MarvinPluginLoader.loadImagePlugin("org.marvinproject.image.morphological.erosion.jar");
		}

		private static void showSegments(List<MarvinSegment> segments, MarvinImage image){
	        for(MarvinSegment s:segments){
	            image.fillRect((s.x1+s.x2)/2, (s.y1+s.y2)/2, 5, 5, Color.red);
	        }
	    }
	     
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			Path inputPath = new Path(value.toString());
			MarvinImage image = HadoopImageIO.loadImage(fs, inputPath);
			System.out.println(image.getHeight());
			System.out.println(image.getWidth());
			MarvinImage result = image.clone();
			
			thresholding(image, 30);
			HadoopImageIO.save(image, fs, new Path("/tmp/out/"+inputPath.getName()+"_threshold.png" ));
			
			List<MarvinSegment> listSegments = new ArrayList<MarvinSegment>();
			List<MarvinSegment> listSegmentsTmp = new ArrayList<MarvinSegment>();
			MarvinImage binImage = MarvinColorModelConverter.rgbToBinary(image, 127);
			erosion.setAttribute("matrix", MarvinMath.getTrueMatrix(8, 8));
			erosion.process(binImage.clone(), binImage);
			HadoopImageIO.save(binImage, fs, new Path("/tmp/out/"+inputPath.getName()+"_bin.png"));
			
			MarvinImage binImageRGB = MarvinColorModelConverter.binaryToRgb(binImage);
			MarvinSegment[] segments = floodfillSegmentation(binImageRGB);
			for (MarvinSegment s : segments) {
				if (s.area < 300) {
					listSegments.add(s);
				}
			}
			for (MarvinSegment s : listSegments){
				context.write(new Text(inputPath.getName()+" "+((s.x1+s.x2)/2)),  new Text(""+(s.y1+s.y2)/2));
			}
			showSegments(listSegments, binImageRGB);
			HadoopImageIO.save(binImageRGB, fs, new Path("/tmp/out/"+inputPath.getName()+"_center_8.png"));

			listSegments = new ArrayList<MarvinSegment>();
			//binImage = MarvinColorModelConverter.rgbToBinary(image, 127);
			binImage = MarvinColorModelConverter.rgbToBinary(image, 127);

			erosion.setAttribute("matrix", MarvinMath.getTrueMatrix(18, 18));
			erosion.process(binImage.clone(), binImage);

			HadoopImageIO.save(binImage, fs, new Path("/tmp/out/"+inputPath.getName()+"_bin.png"));
			binImageRGB = MarvinColorModelConverter.binaryToRgb(binImage);
			segments = floodfillSegmentation(binImageRGB);

			for (MarvinSegment s : segments) {
				listSegments.add(s);
				listSegmentsTmp.add(s);
			}

			showSegments(listSegmentsTmp, binImageRGB);
			HadoopImageIO.save(binImageRGB, fs, new Path("/tmp/out/"+inputPath.getName()+"_center_18.png"));
			MarvinSegment.segmentMinDistance(listSegments, 10);

			showSegments(listSegments, result);
			HadoopImageIO.save(result, fs, new Path("/tmp/out/"+inputPath.getName()+"_result.png"));
		}
	}

	@Override
	public int run(String[] args) throws Exception {
		Configuration conf = this.getConf();
		Job job = Job.getInstance(conf);
		job.setJarByClass(EdjeJob.class);
		job.setMapperClass(TokenizerMapper.class);
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(Text.class);

		FileSystem fs = FileSystem.get(super.getConf());
		PrintWriter pw = new PrintWriter(new OutputStreamWriter(fs.create((new Path("/tmp/work.txt")))));
		FileStatus[] status_list = fs.listStatus(new Path("/tmp/imagein"));
		if (status_list != null) {
			for (FileStatus status : status_list) {
				pw.write(status.getPath() + "\n");
			}
		}
		pw.close();

		job.setInputFormatClass(NLineInputFormat.class);
		NLineInputFormat.addInputPath(job, new Path("/tmp/work.txt"));
		NLineInputFormat.setNumLinesPerSplit(job, 1);
		FileOutputFormat.setOutputPath(job, new Path("/tmp/out"));
		job.setOutputFormatClass(TextOutputFormat.class);
		return job.waitForCompletion(true) ? 0 : 1;
	}
}
