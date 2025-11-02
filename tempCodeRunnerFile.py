
  train_DataLoader = DataLoader(
      train_data,
      batch_size = 128
  )
  test_DataLoader = DataLoader(
      test_data,
      batch_size = 128
  )
  # !!concept!!
  for image, label in train_DataLoader:
    print(image[0].shape)
    print(label[0].item())
    break

  # !!concept!!
  # len(train_DataLoader)
  # 60000/128

  """#### Device"""

  if torch.cuda.is_available():
    device = "cuda"
  else:
    device ="cpu"


  feed_fwd_net = FeedForward().to(device)

  # feed_fwd_net = FeedForward().to(device)   #...model
  # train_DataLoader
  loss_func = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(feed_fwd_net.parameters(), lr=0.01)
  #device
  epochs = 10
  train(feed_fwd_net, train_DataLoader,loss_func, optimizer, device, epochs)

  torch.save(feed_fwd_net.state_dict(), "feed_fwd_net.pth")
  print("saved")
