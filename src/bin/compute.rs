#![feature(duration_millis_float)]
use std::{collections::HashMap, fs, io::{ErrorKind, Read, Write}, thread::sleep, time::Duration};
use std::time::Instant;
use std::net::{TcpListener, TcpStream};

use lenia_gpu::{FrameTimeAnalyzer, Function, PackageLenia, Shape};
use lenia_gpu::{Channel, Layer, Lenia, DataLenia};

use arrayfire::*;

fn _creator(size: (usize, usize)) -> (Lenia, u8) {
    let sx = 32;
    let sy = 16;
    let mut matrix = randn::<f32>(Dim4::new(&[size.0 as u64 / sy,size.1 as u64 / sx,1,1]));
    matrix = &matrix - 0.8_f32;
    matrix = clamp(&matrix, &0_f32, &1_f32, false);
    matrix = scale(&matrix, sy as f32, sx as f32, size.0 as i64, size.1 as i64, InterpType::NEAREST);

    let mut lenia = Lenia::new(0.1, HashMap::new(), HashMap::new());
    
    let channel = Channel::new(matrix);
    lenia.channels.insert(0, channel);

    let layer = Layer::new(
        Function::new(Shape::GaussianBump, false, vec![0.15, 0.5], true), 
        Function::new(Shape::GaussianBump, true, vec![0.015, 0.15], true), 
        0, 92
    );
    lenia.layers.insert( 0, layer);

    lenia.channels.get_mut(&0).unwrap().weights.insert(0, 1.);
    
    let dirs = fs::read_dir("data/").unwrap().filter(|e| e.is_ok() ).filter(|e| e.as_ref().unwrap().path().is_dir() )
        .map(|e| e.unwrap().path().file_name().unwrap().to_str().unwrap().to_owned().trim().parse::<u8>() ).filter(|e| e.is_ok() )
        .map(|e| e.unwrap() ).collect::<Vec<u8>>();

    let id = (u8::min_value()..=u8::max_value()).into_iter().filter(|v| !dirs.contains(v) ).min().unwrap();

    (lenia, id)
}

fn main() {
    get_available_backends();
    set_device(1);
    info();
    sleep(Duration::from_secs(3));
    
    let window_size: (usize, usize) = (1024, 1024 );

    let mut current_lenia = 0;
    let mut lenia = DataLenia::load(current_lenia as usize);
    lenia.init();

    let win = Window::new(window_size.0 as i32, window_size.1 as i32, "LeniaCore".to_string());
    let listener = TcpListener::bind("127.0.0.1:2137").unwrap();
    let mut buffer = [0; 1024];
    let mut pause = false;
    let mut fta = FrameTimeAnalyzer::new(10);
    let mut now;

    for c in listener.incoming() {
        let c: TcpStream = c.unwrap();
        c.set_nonblocking(true).unwrap();
        
        while !win.is_closed() {
            now = Instant::now();
            win.draw_image(&lenia.img, None);
            if !pause {lenia.evaluate()}
            lenia.generate_image();
            handle_client(&c, &mut buffer, &mut lenia, &mut pause, &mut fta, &mut current_lenia);
            fta.add_frame_time(now.elapsed().as_millis_f32());
        }
        return
    } 
}


fn handle_client(
    mut stream: &TcpStream, buffer: &mut[u8],
    lenia: &mut Lenia, pause: &mut bool,
    fta: &mut FrameTimeAnalyzer,
    lid: &mut u8
) {
    match stream.read(buffer) {
        Ok(0) => {}
        Ok(bytes_read) => {
            let response = match buffer[0] {
                10 => {
                    *pause = !*pause;
                    vec![8]
                }
                11 => {
                    DataLenia::save(*lid as usize, lenia);
                    vec![1]
                }
                12 => {
                    *lenia = DataLenia::load(buffer[1] as usize);
                    *lid = buffer[1];
                    vec![1]
                }
                13 => {
                    (*lenia, *lid) = _creator((2048, 2048));
                    vec![1]
                }
                9 => {
                    vec![*fta.smooth_frame_time() as u8 ]
                }
                7 => {
                    let mut p = bincode::serialize(&PackageLenia::from_lenia(lenia)).unwrap();
                    p.insert(0, *lid);
                    p
                }
                8 => {
                    let p:PackageLenia = bincode::deserialize(&buffer[1..bytes_read]).unwrap();
                    PackageLenia::update_lenia(&p, lenia);
                    vec![1]
                }
                _ => {vec![0]}
            };

            match stream.write_all(&response) {
                Ok(_) => {}
                Err(e) => eprintln!("Error sending response: {}", e),
            }
        }
        Err(ref e) if e.kind() == ErrorKind::WouldBlock => {}
        Err(e) => eprintln!("Error reading from socket: {}", e),
    }
}
