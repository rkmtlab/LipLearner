
import Foundation

import UIKit
import AVFoundation
import ImageIO
import MobileCoreServices
import CoreGraphics

func makeGifImage(imageArray: [UIImage], gifName: String){
    let fileProperties = [kCGImagePropertyGIFDictionary as String: [kCGImagePropertyGIFLoopCount as String: 0]]
    let frameProperties = [kCGImagePropertyGIFDictionary as String:[kCGImagePropertyGIFDelayTime as String :CMTimeGetSeconds(CMTime(value: 1, timescale: 30))]]
    
    let documentsFolder = try! FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
    let folderURL = documentsFolder.appendingPathComponent("gif")
    let folderExists = (try? folderURL.checkResourceIsReachable()) ?? false

    if !folderExists {
        try! FileManager.default.createDirectory(at: folderURL, withIntermediateDirectories: false)
    }
    let fileURL = folderURL.appendingPathComponent("\(gifName).gif")

    
    guard let destination = CGImageDestinationCreateWithURL(fileURL as CFURL, kUTTypeGIF, imageArray.count, nil) else {
        print("Failed to create GIF image")
        return
    }
    CGImageDestinationSetProperties(destination,fileProperties as CFDictionary?)
    for image in imageArray{
        let cgiImage = image.resize(to: CGSize(width: 160, height: 160)).cgImage!
        CGImageDestinationAddImage(destination,cgiImage,frameProperties as CFDictionary?)
    }
    if CGImageDestinationFinalize(destination){
        print("Created a GIF file \(gifName)")
    }else{
        print("Failed to create GIF image")
    }
    
}

var commandSet = ["play some music",
                  "take a photo",
                  "get directions to gas station",
                  "turn on focus mode",
                  "open twitter",
                  "turn on the flashlight",
                  "send an email",
                  "set an alarm for 8 am",
                  "what's the weather today",
                  "show today's schedule"]
