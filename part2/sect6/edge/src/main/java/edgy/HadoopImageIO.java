package edgy;

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import marvin.image.MarvinImage;
import marvin.util.MarvinErrorHandler;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Iterator;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.event.IIOReadWarningListener;
import javax.imageio.stream.ImageInputStream;

public class HadoopImageIO {
	public static void save(MarvinImage image, FileSystem fs, Path p) {
		image.update();
		try (FSDataOutputStream l = fs.create(p)) {
			ImageIO.write(image.getBufferedImage(), "png", l);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public static MarvinImage loadImage(FileSystem fs, Path p) {
		try {
			if (!fs.exists(p)) {
				throw new RuntimeException("not found" + p);
			}
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		Iterator<?> l_ittReaders = ImageIO
				.getImageReadersByFormatName(p.getName().substring(p.getName().lastIndexOf(".") + 1));
		ImageReader l_reader = (ImageReader) l_ittReaders.next();

		l_reader.addIIOReadWarningListener(new IIOReadWarningListener() {
			public void warningOccurred(ImageReader source, String warning) {
				MarvinErrorHandler.handleDialog(MarvinErrorHandler.TYPE.BAD_FILE, warning);
			}
		});

		MarvinImage l_marvinImage = null;
		BufferedImage l_bufferedImage = null;
		ImageInputStream l_imageInputStream;
		try {
			l_imageInputStream = ImageIO.createImageInputStream(fs.open(p));
			l_reader.setInput(l_imageInputStream);
			l_bufferedImage = l_reader.read(0);
		} catch (Exception e) {
			throw MarvinErrorHandler.handle(MarvinErrorHandler.TYPE.ERROR_FILE_OPEN, p.toString(), e);
		}
		String l_format = "";
		try {
			l_format = l_reader.getFormatName();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		l_marvinImage = new MarvinImage(l_bufferedImage, l_format);
		return l_marvinImage;
	}
}
